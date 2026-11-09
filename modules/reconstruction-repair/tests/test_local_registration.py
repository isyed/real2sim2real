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
    for weight in (-1., float('nan'), True, 6.):
        with pytest.raises(ValueError):LocalRegistrationConfig(depth_violation_weight=weight).validate()


def test_fresh_pose_refinement_does_not_invent_a_previous_accepted_state():
    from real2sim2real.reconstruction_repair.local_registration import propose_initial
    v,f,p,K,obj,h,z,s=example()
    result=propose_initial(v,f,p,K,obj,h,z,s,config=LocalRegistrationConfig(max_evaluations=1))
    assert result['initial_pose_gates_passed']
    assert not result['previous_accepted_pose_used']
    assert not result['proposal_valid_for_tracking']
    assert result['requires_gpu_revalidation']
    assert not result['temporal_continuity_validated']
    assert not result['full_reference_accepted'] and not result['thresholds_relaxed']
    import json
    json.dumps(result, allow_nan=False)


def test_fresh_pose_refinement_does_not_accept_a_depth_invalid_fit():
    from real2sim2real.reconstruction_repair.local_registration import propose_initial
    v,f,p,K,obj,h,z,s=example()
    result=propose_initial(v,f,p,K,obj,h,z+.3,s,config=LocalRegistrationConfig(max_evaluations=1))
    assert not result['initial_pose_gates_passed'] and 'depth' in result['rejected_for']


def test_translation_only_fit_improves_alignment_and_preserves_rotation_exactly():
    v,f,p,K,obj,h,z,s=example();seed=p.copy();seed[0,3]+=.06
    result=propose(v,f,p,seed,K,obj,h,z,s,
        config=LocalRegistrationConfig(max_evaluations=100,preserve_rotation=True))
    assert result['proposal_valid_for_tracking'] and result['metrics']['iou']>.9
    assert np.array_equal(np.asarray(result['pose_camera'])[:3,:3],p[:3,:3])
    assert result['rotation_preserved_exactly'] and not result['thresholds_relaxed']
    assert result['limits']['step_translation_m']==.08
    seed[:3,:3]=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match='same seed/reference rotation'):
        propose(v,f,p,seed,K,obj,h,z,s,config=LocalRegistrationConfig(preserve_rotation=True))


@pytest.mark.parametrize('value',[1,None,'true'])
def test_translation_only_requires_explicit_opt_in(value):
    with pytest.raises(ValueError,match='boolean'):
        LocalRegistrationConfig(preserve_rotation=value).validate()
