from copy import deepcopy
import numpy as np
import pytest
from real2sim2real.controller_actor_training.approach_profile import configure,position_reward

def config():
    obj=dict(pelvis_position_std=.3,palm_position_std=.2,pelvis_position_weight=1.,palm_position_weight=1.,pelvis_orientation_weight=1.,object_orientation_weight=.1)
    curriculum=dict(thresholds=[4,7,10,30],stages=[dict(objective_weights=dict(pelvis_position=1.,palm_position=1.,pelvis_orientation=1.,object_orientation=.1,force_closure=5.)) for _ in range(4)])
    return dict(objective=obj,env=dict(objective=deepcopy(obj)),curriculum=curriculum,
        training=dict(resume=False,curriculum=deepcopy(curriculum)),task=dict(parquet='reference'),scene=dict(world_count=256),
        action={},observation={},sim={})

def test_fresh_profile_preserves_targets_phases_and_nonapproach_terms():
    c=config();original=deepcopy(c);new,r=configure(c)
    assert c==original and new['objective']==new['env']['objective']
    assert new['training']['curriculum']==new['curriculum']
    assert new['task']==c['task'] and new['scene']==c['scene']
    assert new['objective']['object_orientation_weight']==.1
    assert new['curriculum']['stages'][0]['objective_weights']['force_closure']==5
    assert r['fresh_learner_required'] and not r['physical_improvement_established']

def test_coarse_kernel_has_more_signal_at_large_errors_and_same_optimum():
    assert position_reward(0,.5)==1
    assert position_reward(.5**2,.5)>position_reward(.5**2,.3)
    assert np.all(np.diff(position_reward([0,.1,.5],.4))<0)

@pytest.mark.parametrize('value',[0,-1,float('nan'),True])
def test_invalid_profile_values_rejected(value):
    with pytest.raises(ValueError):configure(config(),palm_std_m=value)

def test_replay_and_inconsistent_config_are_not_silently_reused():
    c=config();c['training']['resume']=True
    with pytest.raises(ValueError):configure(c)

def test_only_exact_profile_checkpoint_may_continue_its_own_training():
    fresh,_=configure(config());resume=config();resume['training']['resume']=True
    result,r=configure(resume,prior_configuration=fresh)
    assert r['matching_profile_resume_verified'] and result['training']['resume']
    with pytest.raises(ValueError):configure(resume,prior_configuration=config())
    fresh['task']['parquet']='different-reference'
    with pytest.raises(ValueError):configure(resume,prior_configuration=fresh)
    c=config();c['env']['objective']['palm_position_std']=.4
    with pytest.raises(ValueError):configure(c)
