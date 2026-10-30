import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.source_clock import sample_poses
from real2sim2real.controller_actor_training.reference_timing import clocks


def test_controller_clock_covers_last_camera_sample_without_speedup():
    source,control,sampled=clocks(858,30.,50.)
    assert len(control)==1430
    assert sampled[-1]==source[-1]
    assert source[-1]==pytest.approx(28.5666666667)
    # Integral clocks must not gain a spurious frame through float rounding.
    assert len(clocks(778,50.,50.)[1])==778


def test_sampling_preserves_physical_error_and_slerps_rotation():
    times=np.array([0.,.02,.04])
    values=np.c_[np.column_stack((times+.2,times*0,times*0)),Rotation.from_euler('z',(times*2)[:,None]).as_quat()]
    samples=np.array([0.,1/30])
    result=sample_poses(values,times,samples)
    np.testing.assert_allclose(result[:,0],samples+.2)
    np.testing.assert_allclose(Rotation.from_quat(result[:,3:]).as_rotvec()[:,2],samples*2)
    with pytest.raises(ValueError,match='within'):
        sample_poses(values,times,[0.,.05])


def test_export_keeps_raw_physics_and_uses_native_reference(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from real2sim2real.controller_actor_training.source_clock import export_source_clock
    times=np.arange(5)/50
    fields=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
    rows=[]
    for i,t in enumerate(times):
        row=dict(environment_id=0,object_body_id=0,object_id=0,object_body_name='object',environment_valid=True,step=i,reference_frame=i,time_s=t)
        for prefix,x in [('achieved',t+.2),('reference',t)]:
            row.update({prefix+'_'+k:v for k,v in zip(fields,[x,0,0,0,0,0,1])})
        rows.append(row)
    table=pa.Table.from_pylist(rows).replace_schema_metadata({b'flash_chord.control_fps':b'50',b'flash_chord.step_count':b'5',b'flash_chord.start_frame':b'0'})
    path=tmp_path/'trajectory.parquet';pq.write_table(table,path);raw=path.read_bytes()
    source=np.arange(3)/30
    reference=SimpleNamespace(num_frames=3,fps=30.,object_body_pos_w=lambda:np.c_[source,source*0,source*0][:,None],
        object_body_quat_w=lambda:np.tile([1.,0,0,0],(3,1,1)))
    export_source_clock(path,reference)
    mapped=pq.read_table(path)
    np.testing.assert_allclose(mapped['time_s'].to_numpy(),source)
    np.testing.assert_allclose(mapped['achieved_position_x'].to_numpy(),source+.2)
    np.testing.assert_allclose(mapped['reference_position_x'].to_numpy(),source)
    assert (tmp_path/'control_trajectory.parquet').read_bytes()==raw


def test_reference_timing_accepts_absent_per_hand_fields_and_keeps_endpoints():
    from real2sim2real.controller_actor_training.reference_timing import resample_row
    q=np.tile([1.,0,0,0],(3,1));pos=np.array([[0.,0,0],[.1,0,0],[.2,0,0]])
    row=dict(fps=30.,robot_root_position=pos.tolist(),robot_root_wxyz=q.tolist(),
        robot_joint_positions=np.zeros((3,43)).tolist(),object_body_wxyz=q[:,None].tolist(),
        ee_pose_w=np.concatenate((pos[:,None],q[:,None]),axis=-1).tolist(),
        hand_contact_active=[[False,True,True],[False,False,True]])
    for key in ['hand_frames_w','hand_finger_joints','hand_link_contact_positions','hand_link_contact_normals',
                'hand_object_contact_positions','hand_object_contact_normals','hand_object_contact_part_ids']:
        row[key]=[None,None]
    result,mapping=resample_row(row)
    np.testing.assert_allclose(np.array(result['robot_root_position'])[[0,-1]],pos[[0,-1]])
    assert mapping['sampled_source_seconds'][-1]==pytest.approx(2/30)
    row['hand_frames_w']=[[[1.]],None]
    with pytest.raises(ValueError,match='Explicit timing'):
        resample_row(row)
