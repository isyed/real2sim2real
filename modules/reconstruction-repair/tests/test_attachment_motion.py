import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.attachment_motion import carrier_frames,diagnose,diagnose_point,bridge_positions,regularize_inferred_positions


def scene():
    points=np.array([[-.1,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    c=np.tile(np.eye(4),(6,1,1));c[:,:3,:3]=Rotation.from_euler('z',np.arange(6)[:,None]*5,degrees=True).as_matrix();c[:,:3,3]=np.c_[np.arange(6)*.02,np.ones(6),np.ones(6)*2]
    clouds=np.einsum('tij,pj->tpi',c[:,:3,:3],points)+c[:,None,:3,3]
    offset=np.eye(4);offset[:3,3]=[0,.2,0];objects=c@offset
    return clouds,objects


def test_rigid_contact_model_recovers_motion_without_claiming_hidden_measurements():
    points,objects=scene();frames,quality=carrier_frames(points)
    result,report=diagnose(frames,dict(enumerate(objects)),fit_frames=[0,1,2],heldout_frames=[3,5],attachment_evidence='Synthetic rigid attachment')
    np.testing.assert_allclose(result,objects,atol=1e-10)
    assert quality['maximum_rms_m']<1e-10 and report['anchor_consistency_passed']
    assert not report['reference_approved'] and not report['hidden_motion_measured']


def test_independent_anchor_rejects_attachment_without_refitting_it():
    points,objects=scene();frames,_=carrier_frames(points);obs=dict(enumerate(objects));obs[5]=obs[5].copy();obs[5][0,3]+=.3
    _,report=diagnose(frames,obs,fit_frames=[0,1,2],heldout_frames=[3,5],attachment_evidence='Synthetic candidate')
    assert not report['anchor_consistency_passed'] and not report['heldout_used_for_fitting']
    assert report['residuals'][-1]['translation_residual_m']>.29


def test_borrowed_seed_rotation_cannot_become_an_independent_rotation_veto():
    points,objects=scene();frames,_=carrier_frames(points);obs=dict(enumerate(objects));obs[5]=obs[5].copy();obs[5][:3,:3]=Rotation.from_rotvec([0,0,2.]).as_matrix()
    _,report=diagnose(frames,obs,fit_frames=[0,1,2],heldout_frames=[3,5],attachment_evidence='Last anchor has independent position only',rotation_observed_frames=[0,1,2,3])
    assert report['anchor_consistency_passed'] and report['unvalidated_rotation_frames']==[5]
    assert report['residuals'][-1]['rotation_residual_deg'] is None
    assert not report['residuals'][-1]['rotation_independently_checked']


def test_degenerate_points_and_overlapping_fit_validation_are_rejected():
    with pytest.raises(ValueError):carrier_frames(np.zeros((5,4,3)))
    points,objects=scene();frames,_=carrier_frames(points)
    with pytest.raises(ValueError):diagnose(frames,dict(enumerate(objects)),fit_frames=[0,1,2],heldout_frames=[2,5],attachment_evidence='x')


def test_deformation_is_reported_and_never_fitted_as_scale():
    points,_=scene();points[-1]*=1.5;_,quality=carrier_frames(points)
    assert quality['maximum_rms_m']>.01 and 5 not in quality['valid_frames']
    assert not quality['scale_fitted']


def test_point_attachment_does_not_tie_object_rotation_to_wrist_rotation():
    points,objects=scene();frames,_=carrier_frames(points)
    positions,report=diagnose_point(frames,{i:p[:3,3] for i,p in enumerate(objects)},fit_frames=[0,1,2],heldout_frames=[3,5],attachment_evidence='Point contact only')
    np.testing.assert_allclose(positions,objects[:,:3,3],atol=1e-10)
    assert report['anchor_consistency_passed'] and not report['object_orientation_estimated']


def test_bridge_has_exact_endpoints_and_retains_inferred_status():
    p=np.c_[np.arange(10)*.01,np.zeros(10),np.ones(10)]
    a=p[2]+[.01,0,0];b=p[8]+[.02,0,0]
    out,r=bridge_positions(p,first_frame=2,last_frame=8,first_position=a,last_position=b)
    np.testing.assert_array_equal(out[0],a);np.testing.assert_array_equal(out[-1],b)
    assert r['passed_numeric_bounds'] and not r['hidden_motion_measured'] and not r['reference_approved']
    assert r['inferred_frame_ids']==list(range(3,8))


def test_large_endpoint_correction_cannot_be_hidden_by_a_smooth_bridge():
    p=np.zeros((10,3));_,r=bridge_positions(p,first_frame=0,last_frame=9,first_position=[0,0,0],last_position=[.3,0,0])
    assert not r['passed_numeric_bounds'] and 'endpoint_correction_bound' in r['reasons']


def test_inferred_smoothing_preserves_anchors_and_correction_bound():
    p=np.c_[np.arange(20)*.01,np.zeros(20),np.ones(20)];p[10,1]=.06
    q,r=regularize_inferred_positions(p)
    np.testing.assert_array_equal(q[[0,-1]],p[[0,-1]])
    assert np.linalg.norm(q-p,axis=1).max()<=.030000001 and r['passed_numeric_bounds']
    assert not r['source_observations_modified'] and not r['reference_approved']


def test_huge_carrier_jump_still_fails_after_bounded_smoothing():
    p=np.zeros((20,3));p[10,0]=.4;_,r=regularize_inferred_positions(p)
    assert not r['passed_numeric_bounds'] and r['maximum_step_m']>.3
