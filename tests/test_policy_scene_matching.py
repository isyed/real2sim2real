from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training.policy_physics import match_scene,match_contact_mode


def configs():
    sim=dict(solver='newton',integrator='implicitfast',cone='pyramidal',impratio=1.,iterations=100,ls_iterations=50,fps=50.,substeps=4)
    training=dict(scene=dict(world_count=256,contact_friction=1.,object_free_joint_damping=0.),sim=sim,env=dict(sim=deepcopy(sim)),objective=dict(orientation=.1),curriculum=dict(thresholds=[2,4]))
    policy=dict(scene=dict(world_count=1,contact_friction=1.,object_free_joint_damping=.01),sim=deepcopy(sim),env=dict(sim=deepcopy(sim)))
    return training,policy


def test_explicit_matching_preserves_training_parallelism_rewards_and_input():
    training,policy=configs();original=deepcopy(training)
    result,receipt=match_scene(training,policy)
    assert training==original and result['scene']['world_count']==256
    assert result['scene']['object_free_joint_damping']==.01
    assert result['objective']==original['objective'] and result['curriculum']==original['curriculum']
    assert receipt['changed_fields']==['object_free_joint_damping']


def test_solver_difference_is_not_silently_copied():
    training,policy=configs();policy['sim']['impratio']=20.
    with pytest.raises(ValueError,match='Solver mismatch'):match_scene(training,policy)


def test_missing_or_invalid_materials_are_rejected():
    training,policy=configs();policy['scene']['contact_friction']=float('nan')
    with pytest.raises(ValueError):match_scene(training,policy)


def test_contact_mode_adoption_is_explicit_and_only_changes_the_boolean():
    training,policy=configs();before=deepcopy(training)
    policy['sim']['enable_multiccd']=policy['env']['sim']['enable_multiccd']=True
    with pytest.raises(ValueError,match='Contact-mode mismatch'):match_scene(training,policy)
    adopted,receipt=match_contact_mode(training,policy)
    assert training==before and receipt['changed'] and not receipt['resume_compatibility_established']
    assert adopted['sim']['enable_multiccd'] is True and adopted['env']['sim']['enable_multiccd'] is True
    adopted['sim'].pop('enable_multiccd');adopted['env']['sim'].pop('enable_multiccd')
    assert adopted==before
    matched,_=match_scene(match_contact_mode(training,policy)[0],policy)
    assert matched['scene']['world_count']==256


@pytest.mark.parametrize('kind',['policy_disagrees','training_disagrees','nonboolean'])
def test_inconsistent_contact_modes_are_rejected(kind):
    training,policy=configs()
    if kind=='policy_disagrees':policy['sim']['enable_multiccd']=True
    if kind=='training_disagrees':training['sim']['enable_multiccd']=True
    if kind=='nonboolean':policy['sim']['enable_multiccd']='false'
    with pytest.raises(ValueError):match_contact_mode(training,policy)
