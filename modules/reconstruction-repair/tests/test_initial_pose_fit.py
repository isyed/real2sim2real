import numpy as np
import pytest
from real2sim2real.reconstruction_repair.initial_pose_fit import fit,InitialFitConfig


def scene():
    x,y=np.meshgrid(np.linspace(-.2,.2,30),np.linspace(-.2,.2,30));vertices=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    faces=np.array([[j*30+i,j*30+i+1,(j+1)*30+i] for j in range(29) for i in range(29)]+[[j*30+i+1,(j+1)*30+i+1,(j+1)*30+i] for j in range(29) for i in range(29)]);pose=np.eye(4);pose[2,3]=2
    K=np.array([[100.,0,32],[0,100,32],[0,0,1]])
    target=np.zeros((64,64),bool);target[22:43,22:43]=True
    support=dict(plane_camera=[0,-1,0,1],center_camera_m=[0,1,2],radius_m=1,coordinate_frame='camera',ground_truth_used=False,evidence='own predicted floor')
    return vertices,faces,pose[None],K,target,np.zeros_like(target),np.full((64,64),2.),support


def test_first_frame_needs_no_invented_temporal_reference_and_cannot_approve_trajectory():
    result=fit(*scene(),config=InitialFitConfig(max_evaluations=1,max_seeds=1))
    assert result['proposal_valid_on_cpu'] and result['temporal_reference'] is None
    assert result['requires_independent_gpu_validation'] and not result['full_reference_accepted']
    assert result['evaluations']==1 and not result['thresholds_relaxed']


def test_bad_depth_is_not_bypassed_by_good_outline():
    args=list(scene());args[6]=np.full((64,64),3.)
    result=fit(*args,config=InitialFitConfig(max_evaluations=1,max_seeds=1))
    assert not result['proposal_valid_on_cpu'] and 'depth'in result['selected']['rejected_for']


def test_invalid_provenance_and_work_bounds_fail():
    args=list(scene());args[-1]=dict(args[-1],ground_truth_used=True)
    with pytest.raises(ValueError):fit(*args)
    with pytest.raises(ValueError):InitialFitConfig(max_evaluations=True).validate()


def test_valid_measured_candidate_is_retained_over_better_cost_invalid_hypothesis(monkeypatch):
    from real2sim2real.reconstruction_repair import initial_pose_fit as module
    args=list(scene());args[2]=args[2].copy();args[2][0,0,3]=.1
    original=module.consistency
    def geometry(points,pose,*rest):
        value=original(points,pose,*rest);value['depth_error_m']=.101 if abs(pose[0,3])<.01 else .09
        return value
    def optimizer(objective,*args,**kwargs):objective(np.array([-.1,0,0,0,0,0]))
    monkeypatch.setattr(module,'consistency',geometry);monkeypatch.setattr(module,'minimize',optimizer)
    result=module.fit(*args,config=InitialFitConfig(max_evaluations=2,max_seeds=1))
    assert result['proposal_valid_on_cpu'] and not result['selected']['rejected_for']
    assert result['selected']['pose_camera'][0][3]==pytest.approx(.1)
    assert any('depth'in r['rejected_for'] for r in result['all_candidates'])
