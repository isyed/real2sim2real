import pytest
from real2sim2real.reconstruction.tracking_recovery import continuous_work_status


def test_exhaustion_is_distinct_from_an_attempt_on_the_failing_frame():
    value=continuous_work_status(used=8,limit=8,frame_attempts=0)
    assert value['budget_exhausted'] and value['no_cpu_attempt_on_frame']
    assert value['remaining_calls']==0 and not value['quality_limits_changed']
    assert not value['additional_work_would_recover_pose_established']


def test_record_attempts_without_claiming_search_is_exhaustive():
    value=continuous_work_status(used=3,limit=8,frame_attempts=2)
    assert not value['budget_exhausted'] and value['remaining_calls']==5
    assert not value['no_cpu_attempt_on_frame']


@pytest.mark.parametrize('used,limit,attempts',[(9,8,0),(1,8,2),(-1,8,0),(0,0,0),(True,8,0)])
def test_invalid_counts_fail_closed(used,limit,attempts):
    with pytest.raises(ValueError):continuous_work_status(used=used,limit=limit,frame_attempts=attempts)
