import numpy as np
import pytest
from real2sim2real.controller_actor_training.approach_diagnostics import summarize,continuation


def trial():
    q=np.zeros((101,3,7));q[:,:,6]=1;q[:,0,2]=.8
    return q,q.copy(),np.zeros((101,2)),np.array([0,1]),np.ones((101,2),bool)


def test_fallen_robot_contact_does_not_count_toward_continuation():
    q,target,contacts,slots,phase=trial();q[40:,0,2]=.2;contacts[40:,0]=1
    result=summarize(q,target,contacts,slots,phase,pelvis_body=0,palm_bodies=[1,2],fps=50)
    assert result['longest_hand_contact_s']>1 and result['upright_intended_contact_s']==0
    assert result['survival_s']==.8


def test_contact_uses_the_matching_hand_phase_and_stops_at_first_balance_loss():
    q,target,contacts,slots,phase=trial();phase[:,0]=False;contacts[10:30,0]=1;contacts[30:50,1]=1
    q[40:45,0,2]=.2
    result=summarize(q,target,contacts,slots,phase,pelvis_body=0,palm_bodies=[1,2],fps=50)
    assert result['upright_intended_contact_s']==.2
    assert result['upright_intended_contact_frames_lr']==[0,10]


def test_full_window_duration_uses_elapsed_time_not_an_extra_control_step():
    result=summarize(*trial(),pelvis_body=0,palm_bodies=[1,2],fps=50)
    assert result['window_s']==2 and result['survival_s']==2 and result['no_low_pelvis_event']


def test_incomplete_phase_or_nonfinite_pose_cannot_qualify():
    q,target,contacts,slots,phase=trial()
    with pytest.raises(ValueError):summarize(q,target,contacts,slots,phase[:-1],pelvis_body=0,palm_bodies=[1,2],fps=50)
    q[0,0,0]=np.nan
    with pytest.raises(ValueError):summarize(q,target,contacts,slots,phase,pelvis_body=0,palm_bodies=[1,2],fps=50)


def test_contact_gain_cannot_hide_a_failing_repeat():
    old=dict(survival_s=7.,min_survival_s=7.,upright_intended_contact_s=0.,upright_palm_rmse_first4s_m=.2,auc=.3,valid=True)
    new=dict(old,survival_s=7.2,min_survival_s=4.,upright_intended_contact_s=2.)
    assert not continuation(old,new,frozen_auc=.3)['continue_to_10m']
    new.update(survival_s=8.,min_survival_s=8.)
    assert continuation(old,new,frozen_auc=.3)['continue_to_10m']


def test_auc_gain_alone_is_insufficient_and_bad_reference_tracking_is_rejected():
    old=dict(survival_s=7.,min_survival_s=7.,upright_intended_contact_s=0.,upright_palm_rmse_first4s_m=.2,auc=.3,valid=True)
    assert not continuation(old,dict(old,auc=.8),frozen_auc=.3)['continue_to_10m']
    assert not continuation(old,dict(old,survival_s=8.,min_survival_s=8.,auc=.1),frozen_auc=.3)['continue_to_10m']
