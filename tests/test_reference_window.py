import pytest
from real2sim2real.controller_actor_training.reference_window import slice_row


def row():
    return dict(robot_root_position=[[i,0,1] for i in range(6)],robot_root_wxyz=[[1,0,0,0]]*6,
        robot_joint_positions=[[i]*43 for i in range(6)],ee_pose_w=[[[i]*7]*2 for i in range(6)],
        hand_contact_active=[[0,0,1,1,0,0],[0,1,1,0,0,0]],hand_frames_w=[None,None],fps=30.,sequence_id='example')


def test_window_preserves_exact_values_and_source():
    original=row();changed,proof=slice_row(original,1,4)
    assert changed['robot_root_position']==original['robot_root_position'][1:4]
    assert changed['hand_contact_active']==[[0,1,1],[1,1,0]]
    assert len(original['robot_root_position'])==6 and changed['hand_frames_w']==[None,None]
    assert not proof['full_task_evaluation'] and proof['frames']==3


@pytest.mark.parametrize('start,stop',[(-1,3),(2,3),(0,7),(2,1),(True,4)])
def test_invalid_windows_fail(start,stop):
    with pytest.raises(ValueError):slice_row(row(),start,stop)


def test_mismatched_side_clock_is_not_silently_clipped():
    original=row();original['hand_contact_active'][0]=[0]
    with pytest.raises(ValueError):slice_row(original,0,3)


def test_portable_window_excludes_stale_derived_control_clock(tmp_path):
    import pyarrow as pa,pyarrow.parquet as pq
    from real2sim2real.artifacts import publish,load
    from real2sim2real.controller_actor_training.reference_window import prepare
    source=tmp_path/'source';source.mkdir();pq.write_table(pa.Table.from_pylist([row()]),source/'reference.parquet')
    publish(source,'robot-reference',{'reference':source/'reference.parquet'},metadata={'robot':'g1','frames':6})
    (source/'control_reference.parquet').write_text('old derived clock')
    out=tmp_path/'window';prepare(source,out,start=1,stop=4,hardlink=True)
    assert not (out/'control_reference.parquet').exists()
    assert load(out)['metadata']['frames']==3
    assert len(pq.read_table(source/'reference.parquet').to_pylist()[0]['robot_root_position'])==6
