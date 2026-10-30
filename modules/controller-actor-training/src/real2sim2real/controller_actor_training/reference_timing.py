"""Preserve source duration when camera and controller clocks differ.

Joint coordinates interpolate linearly, orientations use SLERP, labels use
nearest timestamps, and palm poses are recomputed from robot FK. This preserves
the source clip duration without changing the controller's 50 Hz frequency.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def clocks(frames, source_fps, control_fps):
    if frames < 2 or not np.isfinite([source_fps, control_fps]).all() or min(source_fps, control_fps) <= 0:
        raise ValueError('Invalid reference clock')
    source = np.arange(frames)/source_fps
    control = np.arange(int(np.ceil(source[-1]*control_fps - 1e-9))+1)/control_fps
    return source, control, np.minimum(control, source[-1])


def interpolate(values, source, samples, *, discrete=False):
    values = np.asarray(values)
    if discrete:
        right = np.clip(np.searchsorted(source, samples), 0, len(source)-1)
        left = np.maximum(right-1, 0)
        choose = np.where(samples-source[left] <= source[right]-samples, left, right)
        return values[choose]
    flat = values.reshape(len(source), -1)
    return np.column_stack([np.interp(samples, source, x) for x in flat.T]).reshape((len(samples),)+values.shape[1:])


def rotations(values, source, samples):
    values = np.asarray(values)
    flat = values.reshape(len(source), -1, 4)
    result = [Slerp(source, Rotation.from_quat(track[:, [1, 2, 3, 0]]))(samples).as_quat()[:, [3, 0, 1, 2]]
              for track in flat.transpose(1, 0, 2)]
    return np.stack(result, axis=1).reshape((len(samples),)+values.shape[1:])


def resample_row(original, control_fps=50.):
    """Resample motion_v1; FK palm rebuilding and validation follow separately."""
    row = copy.deepcopy(original)
    frames = len(row['robot_root_position'])
    source, control, samples = clocks(frames, row['fps'], control_fps)
    continuous = ('robot_root_position', 'robot_joint_positions', 'object_body_position',
                  'object_articulation', 'object_root_position')
    for key in continuous:
        if key in row:
            row[key] = interpolate(row[key], source, samples).tolist()
    for key in ('robot_root_wxyz', 'object_body_wxyz'):
        row[key] = rotations(row[key], source, samples).tolist()
    pose = np.asarray(row['ee_pose_w'])
    row['ee_pose_w'] = np.concatenate((interpolate(pose[..., :3], source, samples),
        rotations(pose[..., 3:], source, samples)), axis=-1).tolist()
    # This export currently has no hand geometry arrays beyond the two palms.
    # Fail instead of silently preserving a future export's incompatible clock.
    for key in ('hand_frames_w', 'hand_finger_joints', 'hand_link_contact_positions',
                'hand_link_contact_normals', 'hand_object_contact_positions',
                'hand_object_contact_normals', 'hand_object_contact_part_ids'):
        if any(value is not None for value in np.asarray(row.get(key, []), dtype=object).flat):
            raise ValueError(f'Explicit timing support required for nonempty {key}')
    row['hand_contact_active'] = interpolate(np.asarray(row['hand_contact_active']).T,
        source, samples, discrete=True).T.tolist()
    if 'object_root_axis_angle' in row:
        q = np.asarray(row['object_body_wxyz'])[:, 0]
        row['object_root_axis_angle'] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec().tolist()
    # Original IK diagnostics describe source samples, not interpolated samples.
    for key in ('ik_error_per_frame', 'ik_num_iterations', 'frame_task_errors'):
        row.pop(key, None)
    row['fps'] = control_fps
    return row, dict(source_seconds=source, control_seconds=control,
                     sampled_source_seconds=samples, source_frame_coordinates=samples*original['fps'])


def prepare(source, output, *, control_fps=50.):
    """Create a control-rate reference and validate rebuilt G1 palm FK on CPU."""
    import os
    from pathlib import Path
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Reference preparation requires explicit CPU-only execution')
    import pyarrow as pa
    import pyarrow.parquet as pq
    import newton
    import warp as wp
    from flash_chord.data import load_reference
    from flash_chord.embodiments.g1_dex3 import G1Dex3, G1Dex3Config
    from real2sim2real.artifacts import digest,write_json
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    table=pq.read_table(source)
    if table.num_rows!=1:raise ValueError('Expected one motion_v1 sequence')
    original=table.to_pylist()[0];row,mapping=resample_row(original,control_fps)
    schema=pa.schema([field for field in table.schema if field.name in row],metadata=table.schema.metadata)
    pending=output.with_suffix('.pending.parquet')
    if pending.exists():raise FileExistsError(pending)
    pq.write_table(pa.Table.from_pylist([row],schema=schema),pending)
    wp.init();wp.set_device('cpu')
    # Derived palm trajectories must follow the interpolated joints exactly.
    # This temporary binding obtains FK; final binding retains strict validation.
    robot=G1Dex3(G1Dex3Config(validate_palm_frames=False))
    builder=newton.ModelBuilder();layout=robot.build(builder)
    loaded=load_reference(pending,control_fps=control_fps,source_frame_playback=True)
    binding=robot.bind_reference(builder,layout,loaded)
    poses=np.asarray(row['ee_pose_w'])
    for side in ('left','right'):
        i=row['ee_link_names'].index(side+'_hand_palm_link');hand=binding.hand(side)
        poses[:,i,:3]=hand.wrist_pos_w;poses[:,i,3:]=hand.wrist_quat_w
    row['ee_pose_w']=poses.tolist()
    pq.write_table(pa.Table.from_pylist([row],schema=schema),pending)
    checked=load_reference(pending,control_fps=control_fps,source_frame_playback=True)
    G1Dex3().bind_reference(builder,layout,checked)
    native=load_reference(source)
    for getter in ('robot_root_pos_w','robot_joint_pos','object_body_pos_w'):
        np.testing.assert_allclose(getattr(checked,getter)()[[0,-1]],getattr(native,getter)()[[0,-1]],atol=1e-7)
    pending.rename(output)
    np.savez_compressed(output.with_suffix('.timing.npz'),**mapping)
    receipt=dict(source_sha256=digest(source),control_reference_sha256=digest(output),
        source_frames=native.num_frames,source_fps=native.fps,control_frames=checked.num_frames,
        control_fps=control_fps,source_duration_s=float(mapping['source_seconds'][-1]),
        physical_execution=False,palm_fk_validation=True,source_reference_modified=False)
    write_json(output.with_suffix('.timing.json'),receipt)
    return receipt
