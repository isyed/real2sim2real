import pytest
from real2sim2real.reconstruction_repair.foreground_depth import audit_probe


def row(frame, full, crop):
    def variant(z):
        return dict(accepted=True,object_depth_quantiles_m=[] if z is None else [z-.01,z,z+.01])
    return dict(source_frame=frame,variants=dict(full=variant(full),crop=variant(crop)))


def test_floor_pass_and_first_transition_improvement_do_not_hide_later_jump():
    result=audit_probe(dict(rows=[row(20,3,3),row(21,4,3.01),row(22,4,4)]))
    assert result['variants']['crop']['floor_pass_count']==3
    assert result['variants']['crop']['adjacent_jump_count']==1
    assert not result['variants']['crop']['adjacent_native_pairs'][0]['exceeds_notice']
    assert not result['reference_approved'] and result['preferred_variant'] is None


def test_sparse_frames_do_not_count_as_adjacent_measurements():
    result=audit_probe(dict(rows=[row(20,3,3),row(30,6,6)]))
    assert result['variants']['crop']['adjacent_native_pairs']==[]
    assert result['unsampled_intervals']==[[21,29]]
    assert not result['continuous_motion_established']


def test_missing_depth_is_explicit_and_not_interpolated():
    result=audit_probe(dict(rows=[row(20,3,3),row(21,3,None),row(22,3,3)]))
    assert result['variants']['crop']['missing_foreground_frames']==[21]
    assert result['variants']['crop']['adjacent_native_pairs']==[]


@pytest.mark.parametrize('rows',[[row(2,3,3),row(2,3,3)],[row(2,3,3),row(1,3,3)],[]])
def test_invalid_native_timeline_rejected(rows):
    with pytest.raises(ValueError):audit_probe(dict(rows=rows))


def test_nonfinite_foreground_is_rejected_even_when_floor_passes():
    with pytest.raises(ValueError):audit_probe(dict(rows=[row(2,float('nan'),3)]))


def test_smooth_disagreeing_predictions_do_not_choose_a_winner():
    result=audit_probe(dict(rows=[row(20,3.,2.7),row(21,3.01,2.71)]),cross_scale_notice_m=.1)
    assert all(v['floor_pass_count']==2 and v['adjacent_jump_count']==0
               for v in result['variants'].values())
    assert len(result['cross_scale_disagreement'])==2
    assert result['preferred_variant'] is None and not result['reference_approved']
