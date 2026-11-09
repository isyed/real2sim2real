from dataclasses import replace
import numpy as np
import pytest
from test_temporal_pose import fixture,CFG
from real2sim2real.reconstruction_repair.anchored_recovery import refine,visible_occluder


def test_registered_endpoints_are_exact_and_inputs_immutable():
    d=fixture();original=d['poses_camera'].copy();anchors=original[[0,-1]].copy()
    result,report=refine(d,d['frame_ids'][[0,-1]],anchors,config=replace(CFG,benefit_mode='observation'))
    assert np.array_equal(result[[0,-1]],anchors)
    assert np.array_equal(d['poses_camera'],original)
    assert not report['full_reference_accepted'] and not report['physics_validated']
    assert report['initialization']['hypothesis_only']


def test_endpoint_registration_does_not_override_bad_intermediate_evidence():
    d=fixture();d['object_mask'][10]=np.roll(d['object_mask'][10],40,axis=1)
    _,r=refine(d,d['frame_ids'][[0,-1]],d['poses_camera'][[0,-1]],config=replace(CFG,max_function_calls=1,benefit_mode='observation'))
    assert not r['decision']['accepted'] and 'image_gate' in r['decision']['reasons']


def test_occluder_cannot_erase_observed_book_or_affect_undeclared_frames():
    obj=np.zeros((3,20,30),bool);obj[:,5:8,5:8]=True;human=np.zeros_like(obj);human[:,0,0]=True
    result=visible_occluder(obj,human,np.array([8,9,10]),[[2,2],[15,2],[15,15],[2,15]],[9])
    assert np.array_equal(result[[0,2]],human[[0,2]])
    assert result[1,4,4] and not result[1,6,6] and result[:,0,0].all()
    with pytest.raises(ValueError):visible_occluder(obj,human,np.arange(3),[[2,2],[15,2],[15,15]],[9])


def test_unbracketed_and_nonuniform_inputs_rejected():
    d=fixture()
    with pytest.raises(ValueError):refine(d,np.array([1,20]),d['poses_camera'][[1,-1]],config=CFG)
    d['timestamps_s'][10]+=.001
    with pytest.raises(ValueError):refine(d,np.array([0,20]),d['poses_camera'][[0,-1]],config=CFG)


def test_actual_missing_frames_are_explicit_and_never_observed_truth():
    d=fixture();d['object_mask'][10:12]=False;d['fit'][10:12]=False
    result,r=refine(d,np.array([0,20]),d['poses_camera'][[0,-1]],
        config=replace(CFG,benefit_mode='observation',max_function_calls=1),
        unobserved_image_ids=[10,11],unobserved_depth_ids=[10,11])
    assert r['unobserved_image_frame_ids']==[10,11]
    assert not r['inferred_frames_verified_by_observation']
    assert np.array_equal(result[[0,-1]],d['poses_camera'][[0,-1]])
    with pytest.raises(ValueError,match='Visible image'):
        refine(d,np.array([0,20]),d['poses_camera'][[0,-1]],config=CFG,
               unobserved_image_ids=[8])
