import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from real2sim2real.reconstruction_repair.outline_pose import propose


def inputs():
    v=np.array([[a,b,c] for a in [-.05,.05] for b in [-.12,.12] for c in [-.03,.03]])
    k=np.array([[300,0,80],[0,300,80],[0,0,1.]])
    truth=Rotation.from_euler('z',12,degrees=True).as_matrix();t=np.array([.008,-.005,1.])
    p=v@truth.T+t;uv=p@k.T;uv=uv[:,:2]/uv[:,2:]
    image=Image.new('1',(160,160));ImageDraw.Draw(image).polygon([tuple(x) for x in uv[ConvexHull(uv).vertices]],fill=1)
    return v,k,np.asarray(image,bool)


def test_outline_proposal_improves_fit_without_changing_scale_or_center_depth():
    v,k,mask=inputs();original=v.copy();r,t,proof=propose(v,np.eye(3),np.array([0,0,1.]),k,mask,max_rotation_deg=25)
    assert np.array_equal(v,original)
    assert proof['support_rmse_after_px']<proof['support_rmse_before_px']*.4
    assert proof['rotation_change_deg']<=25 and proof['center_shift_m']<=.04
    assert abs((r@v.mean(0)+t)[2]-1.)<1e-12
    assert proof['proposal_only'] and not proof['candidate_approved']


def test_insufficient_or_occluded_outline_fails_closed():
    v,k,mask=inputs()
    with pytest.raises(ValueError,match='overlap'):propose(v,np.eye(3),np.array([0,0,1.]),k,mask,human_mask=mask)
    with pytest.raises(ValueError,match='Insufficient'):propose(v,np.eye(3),np.array([0,0,1.]),k,np.zeros_like(mask))
    with pytest.raises(ValueError):propose(v,np.eye(3)*2,np.array([0,0,1.]),k,mask)


def test_camera_depth_fit_is_explicit_and_remains_a_bounded_hypothesis():
    v,k,mask=inputs();r,t,proof=propose(v,np.eye(3),np.array([0,0,1.05]),k,mask,
        max_rotation_deg=25,max_center_shift_m=.12,preserve_camera_center_depth=False)
    assert not proof['camera_center_depth_preserved']
    assert proof['center_shift_m']<=.12
    assert not proof['depth_used_for_fitting'] and not proof['candidate_approved']


def test_observed_path_rejects_a_cheaper_discontinuous_or_unobserved_pose():
    from real2sim2real.reconstruction_repair.outline_pose import select_observed_path
    p=np.tile(np.eye(4),(3,2,1,1));p[:,0,0,3]=[.01,.02,.03];p[:,1,0,3]=[1.,1.,1.]
    chosen,proof=select_observed_path(p,np.array([[1.,0.]]*3),np.ones((3,2),bool),np.eye(4))
    assert np.allclose(chosen[:,0,3],[.01,.02,.03]) and not proof['interpolation_used']
    with pytest.raises(ValueError,match='local frame 1'):
        select_observed_path(p,np.zeros((3,2)),np.array([[True,True],[False,False],[True,True]]),np.eye(4))


def test_correction_limit_prevents_pose_switch_jitter_without_blocking_source_motion():
    from real2sim2real.reconstruction_repair.outline_pose import select_observed_path
    parent=np.tile(np.eye(4),(3,1,1));parent[1:,:3,:3]=Rotation.from_euler('z',29,degrees=True).as_matrix()
    p=np.repeat(parent[:,None],2,axis=1)
    p[:,1,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()@parent[:,:3,:3]
    chosen,proof=select_observed_path(p,np.array([[.1,0.]]*3),np.ones((3,2),bool),np.eye(4),
        reference_poses=parent,max_correction_step_deg=6,correction_weight=.01)
    assert np.array_equal(chosen,parent)
    assert proof['max_correction_step_deg']==6
    with pytest.raises(ValueError):select_observed_path(p,np.zeros((3,2)),np.ones((3,2),bool),np.eye(4),max_correction_step_deg=6)


def test_motion_review_rejects_new_jitter_even_below_per_frame_30_degree_bound():
    from real2sim2real.reconstruction_repair.outline_pose import review_motion
    parent=np.tile(np.eye(4),(20,1,1));noisy=parent.copy()
    noisy[1::2,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()
    result=review_motion(parent,noisy)
    assert not result['passed'] and result['after']['max_rotation_step_deg']<30
    assert review_motion(parent,parent)['passed']
    with pytest.raises(ValueError):review_motion(parent,noisy[:-1])


def test_motion_review_catches_sparse_staircase_jumps_hidden_by_p95():
    from real2sim2real.reconstruction_repair.outline_pose import review_motion
    parent=np.tile(np.eye(4),(600,1,1));staircase=parent.copy()
    parent[:,:3,:3]=Rotation.from_euler('z',np.linspace(0,300,600)[:,None],degrees=True).as_matrix()
    staircase[:,:3,:3]=Rotation.from_euler('z',((np.arange(600)//29)*15)[:,None],degrees=True).as_matrix()
    result=review_motion(parent,staircase)
    assert result['after']['p95_rotation_step_deg']<result['before']['p95_rotation_step_deg']
    assert result['after']['high_rotation_step_count']>5
    assert not result['passed']
