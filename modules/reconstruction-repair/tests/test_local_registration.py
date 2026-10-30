import numpy as np
import pytest
from real2sim2real.reconstruction_repair.local_registration import propose,LocalRegistrationConfig
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette


def example():
    x,y=np.meshgrid(np.linspace(-.3,.3,20),np.linspace(-.3,.3,20))
    vertices=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)];faces=[]
    for row in range(19):
        for col in range(19):
            i=row*20+col;faces += [[i,i+1,i+20],[i+1,i+21,i+20]]
    faces=np.asarray(faces);pose=np.eye(4);pose[2,3]=2
    K=np.array([[80.,0,40],[0,80,40],[0,0,1]])
    g=dict(vertices=vertices,faces=faces,object_scale=np.array(1.),object_rotation=pose[None,:3,:3],
           object_translation=pose[None,:3,3],intrinsics=K[None],world_to_camera=np.eye(4)[None])
    mask=mesh_silhouette(g,0,80,80)
    support=dict(plane_camera=[0.,-1.,0.,2.],center_camera_m=[0,2,0],radius_m=1e6,
                 ground_truth_used=False,evidence='Synthetic predicted support')
    return vertices,faces,pose,K,mask,np.zeros_like(mask),np.full(mask.shape,2.),support


def test_continuous_proposal_improves_small_pose_error_without_changing_limits():
    v,f,p,K,obj,h,z,s=example();initial=p.copy();initial[0,3]+=.06
    before=propose(v,f,p,initial,K,obj,h,z,s,config=LocalRegistrationConfig(max_evaluations=1))
    after=propose(v,f,p,initial,K,obj,h,z,s,config=LocalRegistrationConfig(max_evaluations=100))
    assert after['metrics']['iou']>before['metrics']['iou']+.05
    assert after['proposal_valid_for_tracking'] and after['native_acceptance']
    assert after['limits']==before['limits'] and not after['thresholds_relaxed']
    assert after['evaluations']<=100
    assert not after['full_reference_accepted'] and not after['metric_calibration_approved']


def test_wrong_size_does_not_become_an_accepted_reference_or_implicit_resize():
    v,f,p,K,obj,h,z,s=example()
    result=propose(v*.4,f,p,p,K,obj,h,z,s,config=LocalRegistrationConfig(max_evaluations=1))
    assert not result['proposal_valid_for_tracking'] and 'iou' in result['rejected_for']
    assert not result['full_reference_accepted']


def test_out_of_bounds_seed_and_oracle_support_cannot_enter_recovery():
    v,f,p,K,obj,h,z,s=example();seed=p.copy();seed[0,3]=.2
    with pytest.raises(ValueError,match='motion bounds'):propose(v,f,p,seed,K,obj,h,z,s)
    s['ground_truth_used']=True
    with pytest.raises(ValueError,match='provenance'):propose(v,f,p,p,K,obj,h,z,s)


def test_native_grid_mismatch_and_invalid_deadline_fail():
    v,f,p,K,obj,h,z,s=example()
    with pytest.raises(ValueError,match='native'):propose(v,f,p,p,K,obj,h,z[:-1],s)
    with pytest.raises(ValueError):LocalRegistrationConfig(max_seconds=float('inf')).validate()
