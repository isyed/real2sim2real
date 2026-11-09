import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.source_pose_audit import corresponding_mesh_transform,bind_pose_frames,projection_audit


def test_fixed_mesh_and_world_frames_cancel_without_repairing_poses():
    rng=np.random.default_rng(7);v=rng.normal(size=(100,3))*.1
    s=Rotation.from_euler('xyz',[.2,-.4,.7]).as_matrix();d=np.array([.02,.01,-.03])
    m=corresponding_mesh_transform(v,v@s.T+d)
    assert m['max_residual_m']<1e-12
    rc=Rotation.from_euler('z',np.linspace(0,1,10)[:,None]).as_matrix();tc=rng.normal(size=(10,3))
    a=Rotation.from_euler('x',.3).as_matrix();b=np.array([1.,2.,3.])
    rw=a@rc@s.T;tw=tc@a.T+b-np.einsum('tij,j->ti',rw,d)
    proof=bind_pose_frames(rc,tc,rw,tw,m)
    assert proof['max_position_residual_m']<1e-12 and proof['max_rotation_residual_deg']<1e-10
    tw[4,0]+=.02
    assert bind_pose_frames(rc,tc,rw,tw,m)['position_residual_m'][4]>.019


def test_native_roi_preserves_silhouette_and_prediction_depth():
    v=np.array([[-.6,-.6,2],[.6,-.6,2],[.6,.6,2],[-.6,.6,2]])
    f=np.array([[0,1,2],[0,2,3]]);k=np.array([[20.,0,20],[0,20,20],[0,0,1]])
    mask=np.zeros((40,40),bool);mask[14:27,14:27]=True
    result,surface=projection_audit(v,f,k,mask,np.full((40,40),2.))
    assert result['full_silhouette']['iou']==pytest.approx(1)
    assert result['predicted_surface_depth']['consistency_passed']
    assert not result['candidate_approved']
    result,_=projection_audit(v,f,k,mask,np.full((40,40),2.3))
    assert not result['predicted_surface_depth']['consistency_passed']


def test_human_mask_exclusion_is_not_silent_visibility_truth():
    v=np.array([[-.5,-.5,2],[.5,-.5,2],[0,.5,2]])
    mask=np.ones((40,40),bool);human=np.ones_like(mask)
    result,_=projection_audit(v,np.array([[0,1,2]]),np.array([[20.,0,20],[0,20,20],[0,0,1]]),mask,np.full((40,40),2.),human_mask=human)
    assert result['full_silhouette']['iou'] is not None
    assert result['human_excluded_silhouette']['iou'] is None
    assert not result['predicted_surface_depth']['sufficient_evidence']


def test_different_mesh_or_zero_scale_is_not_a_frame_conversion():
    with pytest.raises(ValueError):corresponding_mesh_transform(np.zeros((3,3)),np.ones((3,3)))
    with pytest.raises(ValueError):projection_audit(np.array([[0,0,-1]]),np.array([[0,0,0]]),np.eye(3),np.ones((3,3),bool),np.ones((3,3)))
