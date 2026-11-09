from copy import deepcopy
import numpy as np
import pytest
from real2sim2real.controller_actor_training.reward_alignment import audit,without_hand_grasp_reward


def configuration():
    objective=dict(force_closure_weight=5.,palm_position_weight=1.,body_joint_position_weight=5.,object_position_weight=1.)
    curriculum=dict(stages=[dict(objective_weights=dict(force_closure=5.,object_position=1.)) for _ in range(4)])
    return dict(objective=objective,env=dict(objective=deepcopy(objective)),curriculum=curriculum,
        collision=dict(robot_scope='all',robot_object_collision=True),
        training=dict(curriculum=deepcopy(curriculum),resume=False,checkpoint_path=None,checkpoint=dict(load_replay_path=None)))


def test_nonhand_task_with_zero_hand_labels_still_has_object_reward():
    result=audit(configuration(),np.zeros((2,20)),expected_effectors=['left_foot'],control_fps=50.)
    assert result['object_outcome_reward_present']
    assert not result['hand_grasp_reward_active']
    assert not result['training_admission_established']
    assert not result['specific_nonhand_contact_reward_implemented']


def test_incidental_hand_phase_is_reported_without_claiming_foot_contact():
    phases=np.zeros((2,20));phases[0,10:12]=1
    result=audit(configuration(),phases,expected_effectors=['left_foot'],control_fps=50.)
    assert result['hand_phase_frames_by_channel']==[2,0]
    assert result['hand_bonus_enabled_seconds']==.04
    assert 'incidental_hand_grasp_reward_on_nonhand_task' in result['issues']
    assert 'hand_contact_diagnostics_do_not_measure_task_effector_contact' in result['issues']


def test_profile_changes_every_runtime_stage_without_mutating_input():
    original=configuration();saved=deepcopy(original)
    changed=without_hand_grasp_reward(original,expected_effectors=['left_foot'])
    assert original==saved
    assert changed['objective']['force_closure_weight']==0
    assert changed['objective']==changed['env']['objective']
    assert changed['curriculum']==changed['training']['curriculum']
    assert all(s['objective_weights']['force_closure']==0 for s in changed['curriculum']['stages'])
    assert changed['objective']['palm_position_weight']==1
    assert changed['objective']['object_position_weight']==1


@pytest.mark.parametrize('field',['resume','checkpoint_path','replay'])
def test_profile_rejects_midrun_reward_changes(field):
    cfg=configuration()
    if field=='replay':cfg['training']['checkpoint']['load_replay_path']='saved_replay'
    else:cfg['training'][field]=True
    with pytest.raises(ValueError,match='fresh learner'):
        without_hand_grasp_reward(cfg,expected_effectors=['left_foot'])


def test_hand_task_or_disabled_nonhand_collisions_cannot_use_profile():
    with pytest.raises(ValueError,match='hand task'):
        without_hand_grasp_reward(configuration(),expected_effectors=['hand'])
    cfg=configuration();cfg['collision']['robot_scope']='contact_only'
    with pytest.raises(ValueError,match='collisions'):
        without_hand_grasp_reward(cfg,expected_effectors=['left_foot'])


def test_hand_task_without_contact_is_reported_and_fractional_labels_rejected():
    result=audit(configuration(),np.zeros((2,20)),expected_effectors=['hand'],control_fps=50.)
    assert 'required_hand_phases_missing' in result['issues']
    with pytest.raises(ValueError,match='binary'):
        audit(configuration(),np.ones((2,20))*.5,expected_effectors=['foot'],control_fps=50.)
