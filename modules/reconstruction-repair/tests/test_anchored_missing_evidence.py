from dataclasses import replace
import numpy as np
import pytest

from test_temporal_pose import fixture,CFG
from real2sim2real.reconstruction_repair.temporal_pose import assessment_masks,refine_window
from real2sim2real.reconstruction_repair.anchor_hypothesis import initialize


def test_anchor_hypothesis_keeps_exact_anchors_but_claims_no_hidden_truth():
    poses=np.tile(np.eye(4),(2,1,1));poses[1,0,3]=.2
    result,receipt=initialize(np.arange(100,111),np.array([100,110]),poses,fps=30.)
    assert np.array_equal(result[[0,-1]],poses)
    assert receipt['hypothesis_only'] and not receipt['accepted_reference']
    assert not receipt['interpolated_frames_verified']
    with pytest.raises(ValueError):initialize(np.arange(100,111),np.array([101,110]),poses,fps=30.)


def test_missing_measurements_require_brackets_and_cannot_hide_present_evidence():
    d=fixture();cfg=replace(CFG,benefit_mode='observation')
    with pytest.raises(ValueError,match='Visible image'):
        assessment_masks(d,cfg,[10],[])
    with pytest.raises(ValueError,match='Available depth'):
        assessment_masks(d,cfg,[],[10])
    d['object_mask'][10]=False;d['fit'][10]=False
    image,depth=assessment_masks(d,cfg,[10],[10])
    assert not image[10] and not depth[10]
    d['locked'][-1]=False
    with pytest.raises(ValueError,match='endpoint'):
        assessment_masks(d,cfg,[10],[10])


def test_inferred_frame_is_reported_without_turning_it_into_an_observation():
    d=fixture();d['object_mask'][10]=False;d['fit'][10]=False
    _,report=refine_window(d,replace(CFG,max_function_calls=1,benefit_mode='observation'),
        unobserved_image_ids=[10],unobserved_depth_ids=[10])
    assert report['unobserved_image_frame_ids']==[10]
    assert not report['inferred_frames_verified_by_observation']
    assert not report['full_trajectory_accepted']


def test_observation_mode_is_opt_in_and_unknown_modes_fail():
    assert CFG.benefit_mode=='temporal'
    with pytest.raises(ValueError,match='Benefit mode'):replace(CFG,benefit_mode='anything').validate()
