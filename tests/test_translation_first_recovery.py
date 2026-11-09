import numpy as np
import pytest
from real2sim2real.reconstruction_repair import local_registration as module


def measurements(pose):
    return dict(pose_camera=pose.tolist(),metrics=dict(iou=.8,precision=.9,recall=.9),
        geometry=dict(depth_points=200,depth_error_m=.03,support_penetration_m=0.,behind_camera=False),
        initial_pose_gates_passed=True,full_reference_accepted=False)


def call(previous,initial):
    return module.propose_translation_first(None,None,previous,initial,None,None,None,None,None)


def test_initial_registration_pass_does_not_authorize_temporal_jump(monkeypatch):
    previous=np.eye(4);initial=np.eye(4);initial[0,3]=.09
    monkeypatch.setattr(module,'propose_initial',lambda *a,**k:measurements(initial))
    monkeypatch.setattr(module,'propose',lambda *a,**k:dict(pose_camera=previous.tolist(),proposal_valid_for_tracking=False))
    result=call(previous,initial)
    assert result['translation_first_attempt']['rejected_for']==['translation_step']
    assert not result['proposal_valid_for_tracking']


def test_fixed_seed_rotation_is_checked_against_real_previous(monkeypatch):
    previous=np.eye(4);initial=np.eye(4);initial[0,3]=.02
    monkeypatch.setattr(module,'propose_initial',lambda *a,**k:measurements(initial))
    def forbidden(*a,**k):raise AssertionError('Passing first stage should not invoke fallback')
    monkeypatch.setattr(module,'propose',forbidden)
    result=call(previous,initial)
    assert result['proposal_valid_for_tracking'] and result['requires_gpu_revalidation']
    assert result['previous_accepted_pose_used'] and not result['full_reference_accepted']


def test_rotation_jump_still_requires_fallback(monkeypatch):
    from scipy.spatial.transform import Rotation
    previous=np.eye(4);initial=np.eye(4);initial[:3,:3]=Rotation.from_euler('y',40,degrees=True).as_matrix()
    monkeypatch.setattr(module,'propose_initial',lambda *a,**k:measurements(initial))
    monkeypatch.setattr(module,'propose',lambda *a,**k:dict(proposal_valid_for_tracking=False))
    assert call(previous,initial)['translation_first_attempt']['rejected_for']==['rotation_step']


def test_bad_temporal_reference_is_rejected():
    with pytest.raises(ValueError):call(np.zeros((4,4)),np.eye(4))
