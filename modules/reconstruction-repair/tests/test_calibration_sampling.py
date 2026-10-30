import numpy as np
import pytest
from real2sim2real.reconstruction_repair.calibration_sampling import select, visibility


def test_occluded_uniform_frame_gets_nearby_input_only_replacement():
    # The midpoint of bin 2 is occluded; tie-breaking is deterministic.
    calls=[]
    def inspect(i, config):
        calls.append(i)
        return dict(eligible=i != 24, reasons=['human_occlusion'] if i == 24 else [])
    result=select(60,inspect,samples=6)
    assert result['selected_indices']==[4,14,25,34,44,54]
    assert result['bins'][2]['split']=='fit'
    assert result['bins'][3]['split']=='held_out'
    assert result['passed'] and result['uses_pose_scale_or_tracking_scores'] is False
    assert calls==[4,14,24,25,34,44,54]


def test_invisible_time_bin_cannot_be_dropped_or_borrow_from_other_bins():
    result=select(60,lambda i,c:dict(eligible=not 20<=i<30,reasons=[]),samples=6)
    assert not result['passed'] and result['missing_bins']==[2]
    assert len(result['selected_indices'])==5


def test_visibility_rejects_occlusion_invalid_depth_and_clipping():
    obj=np.zeros((40,40),bool);obj[5:35,5:35]=True
    person=np.zeros_like(obj);depth=np.ones(obj.shape)
    assert visibility(obj,person,depth)['eligible']
    person[5:25,5:35]=True
    assert 'human_occlusion' in visibility(obj,person,depth)['reasons']
    person[:]=False;depth[:]=0
    assert 'insufficient_depth_pixels' in visibility(obj,person,depth)['reasons']
    depth[:]=1;obj[0,:]=True
    assert 'image_boundary_clipping' in visibility(obj,person,depth)['reasons']


def test_invalid_split_or_native_grids_fail():
    with pytest.raises(ValueError):select(60,lambda i,c:None,samples=7)
    with pytest.raises(ValueError):visibility(np.ones((5,5)),np.ones((6,5)),np.ones((5,5)))


def test_mixed_foreground_and_background_depth_does_not_qualify_for_sampling():
    obj=np.zeros((40,40),bool);obj[5:35,5:35]=True
    depth=np.ones((40,40));depth[20:]=3
    assert 'depth_incoherent' in visibility(obj,np.zeros_like(obj),depth)['reasons']
