"""Sample saved physical object trajectories onto their native reference clock.

The original control-rate file is retained. Positions interpolate linearly and
rotations use SLERP. References come from the unchanged native robot reference,
not from achieved poses. Invalid worlds stay invalid; no target snapping occurs.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from real2sim2real.artifacts import digest, write_json


def sample_poses(values, control_times, source_times):
    values=np.asarray(values,float);control_times=np.asarray(control_times,float);source_times=np.asarray(source_times,float)
    if (values.shape!=(len(control_times),7) or not np.isfinite(values).all()
        or np.any(np.diff(control_times)<=0) or source_times[0]<control_times[0]-1e-8
        or source_times[-1]>control_times[-1]+1e-8):
        raise ValueError('Source timestamps must lie within finite saved physics')
    xyz=np.column_stack([np.interp(source_times,control_times,values[:,i]) for i in range(3)])
    quat=Slerp(control_times,Rotation.from_quat(values[:,3:]))(np.clip(source_times,control_times[0],control_times[-1])).as_quat()
    return np.c_[xyz,quat]


def export_source_clock(path, reference):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=Path(path);table=pq.read_table(path);meta=dict(table.schema.metadata or {})
    count=reference.num_frames;times=np.arange(count)/reference.fps
    control_fps=float(meta[b'flash_chord.control_fps'])
    if int(meta[b'flash_chord.step_count'])==count and np.isclose(control_fps,reference.fps,rtol=1e-8):
        return
    if int(meta[b'flash_chord.start_frame'])!=0:raise ValueError('Source-clock export requires frame-zero execution')
    positions=reference.object_body_pos_w();quaternions=reference.object_body_quat_w()[...,[1,2,3,0]]
    arrays={name:table[name].to_numpy() for name in table.column_names}
    fields=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
    output=[]
    for world in sorted(set(arrays['environment_id'])):
        for body in sorted(set(arrays['object_body_id'])):
            idx=np.flatnonzero((arrays['environment_id']==world)&(arrays['object_body_id']==body))
            clock=arrays['time_s'][idx]
            actual=np.column_stack([arrays['achieved_'+name][idx] for name in fields])
            valid=bool(np.all(arrays['environment_valid'][idx]))
            if not valid:raise ValueError('Invalid physics cannot be exported as a valid baseline')
            sampled=sample_poses(actual,clock,times)
            wanted=np.c_[positions[:,int(body)],quaternions[:,int(body)]]
            for frame,t in enumerate(times):
                row={k:arrays[k][idx[0]] for k in table.column_names}
                row.update(step=frame,reference_frame=frame,time_s=t,environment_valid=valid)
                row.update({'achieved_'+name:sampled[frame,i] for i,name in enumerate(fields)})
                row.update({'reference_'+name:wanted[frame,i] for i,name in enumerate(fields)})
                output.append(row)
    output.sort(key=lambda row:(row['step'],row['environment_id'],row['object_body_id']))
    raw=path.with_name('control_trajectory.parquet')
    if raw.exists():raise FileExistsError(raw)
    meta.update({b'flash_chord.control_fps':str(reference.fps).encode(),b'flash_chord.step_count':str(count).encode(),
        b'flash_chord.simulation_control_fps':str(control_fps).encode(),
        b'flash_chord.timestamp_mapping':b'saved physics -> native source timestamps; linear position and SLERP',
        b'flash_chord.control_trajectory_sha256':digest(path).encode()})
    mapped=pa.Table.from_pylist(output,schema=table.schema.with_metadata(meta))
    temporary=path.with_name('source_clock.partial.parquet');pq.write_table(mapped,temporary,compression='zstd')
    path.rename(raw);temporary.rename(path)
    write_json(path.with_name('source_clock.json'),dict(source_frames=count,source_fps=reference.fps,control_fps=control_fps,
        control_frames=int(table.schema.metadata[b'flash_chord.step_count']),source_duration_s=float(times[-1]),
        raw_sha256=digest(raw),mapped_sha256=digest(path),target_snapping=False))
