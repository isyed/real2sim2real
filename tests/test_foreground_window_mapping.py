import numpy as np
import pytest
from real2sim2real.reconstruction_repair.foreground_visibility import measure,audit,depth_origin

def rows(first):
    m=np.ones((20,20),bool);h=np.zeros_like(m)
    result=[measure(np.ones(m.shape)*z,m,h,frame_id=first+i) for i,z in enumerate([2.,3.,3.,3.])]
    result[2].update(observation_available=False,depth_pixels=0,median_depth_m=None,relative_depth_iqr=None,reasons=['insufficient_visible_depth'])
    return result

def test_window_reports_warnings_and_gaps_in_native_frame_ids():
    r=audit(rows(40),np.arange(4)/30)
    assert r['source_frame_indices']==[40,41,42,43]
    assert r['temporal_depth_warning_frames']==[41]
    gap=r['unavailable_observation_spans'][0]
    assert (gap['first_frame'],gap['last_frame'],gap['last_observed_frame'],gap['reacquisition_frame'])==(42,42,41,43)
    assert r['fresh_registration_needed_after_gaps']==[43]

def test_zero_based_reports_keep_existing_frame_meaning():
    r=audit(rows(0),np.arange(4)/30)
    assert r['temporal_depth_warning_frames']==[1]
    assert r['unavailable_observation_spans'][0]['first_frame']==2

def test_equal_length_different_raw_aligned_windows_rejected():
    with pytest.raises(ValueError,match='coverage differs'):
        depth_origin(rows(40),rows(41),np.arange(4)/30)

def test_sparse_window_does_not_become_contiguous_by_relabeling():
    r=rows(40);r[-1]['frame_id']=45
    with pytest.raises(ValueError,match='consecutive source'):audit(r,np.arange(4)/30)
