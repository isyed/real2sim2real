from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training import frozen, residual


@pytest.fixture
def portable(monkeypatch,tmp_path):
    c=frozen.configuration();c['task']['parquet']=str(tmp_path/'control.parquet')
    c['action']['policy_dir']=str(tmp_path/'assets/policies/sonic');c['env']['action']=deepcopy(c['action'])
    monkeypatch.setattr(frozen,'materialize',lambda p:({},deepcopy(c)))
    return tmp_path,c


def test_training_preserves_controller_but_restores_learning_and_assistance(portable):
    pytest.importorskip("newton", reason="Instantiates optional simulator configuration")
    path,base=portable;c=residual.configuration(path,path/'train',environment_steps=10000000)
    checks=residual.validate(c)
    assert c['action']==base['action'] and c['task']==base['task']
    assert c['training']['total_environment_steps']==9999872
    assert checks['assistance_scales']==[1,.5,.1,0]
    assert all(w['object_orientation']==.1 for w in checks['stage_weights'])
    assert checks['worlds']==256 and checks['replay_capacity']==1000000
    assert not checks['resume'] and c['training']['checkpoint_path'] is None
    assert c['env']['voc']['max_force']>0 and c['env']['auto_reset']


def test_continuation_preserves_reward_network_clock_and_lr_schedule(portable):
    path,_=portable
    a=residual.configuration(path,path/'train',environment_steps=10000000)
    b=residual.configuration(path,path/'train',environment_steps=15000000,resume_state='state',resume_replay='replay')
    for key in ['action','objective','observation','curriculum','task','env']:
        assert a[key]==b[key]
    assert a['training']['optimization']==b['training']['optimization']
    assert b['training']['checkpoint']['load_replay_path']=='replay' and b['training']['resume']


def test_delayed_assistance_ends_at_rounded_ten_million_checkpoint(portable):
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.flash_sac.config import TrainingConfig
    from omegaconf import OmegaConf
    path,_=portable
    c=residual.configuration(path,path/'train',environment_steps=10000000,
        assistance_thresholds=[4000000,7000000,10000000,30000000])
    training=instantiate_typed(OmegaConf.create(c['training']),TrainingConfig)
    assert training.curriculum.stage_at(4999936).voc_scale==.5
    assert training.curriculum.stage_at(9999871).voc_scale==.1
    assert training.curriculum.stage_at(9999872).voc_scale==0
    assert training.curriculum.stage_at(30000000).voc_scale==0
    assert c['curriculum']==c['training']['curriculum']


@pytest.mark.parametrize('thresholds',[[1,2,3,4],[4000000,3000000,10000000,30000000],[4000000,7000000],
    [4000000,4000001,10000000,30000000]])
def test_invalid_custom_assistance_thresholds_rejected(portable,thresholds):
    path,_=portable
    with pytest.raises(ValueError):
        residual.configuration(path,path/'train',environment_steps=10000000,assistance_thresholds=thresholds)


@pytest.mark.parametrize('kwargs',[dict(environment_steps=True),dict(environment_steps=255),dict(environment_steps=1000,worlds=0),dict(environment_steps=1000,resume_state='only')])
def test_invalid_or_partial_resume_fails_closed(portable,kwargs):
    path,_=portable
    with pytest.raises(ValueError):residual.configuration(path,path/'train',**kwargs)


def test_explicit_nonhand_profile_has_zero_grasp_weight_in_all_runtime_stages(portable):
    pytest.importorskip("newton", reason="Instantiates optional simulator configuration")
    from real2sim2real.controller_actor_training.reward_alignment import without_hand_grasp_reward
    path,_=portable
    original=residual.configuration(path,path/'train',environment_steps=5000000)
    changed=without_hand_grasp_reward(original,expected_effectors=['left_foot'])
    checks=residual.validate(changed)
    assert all(stage['force_closure']==0 for stage in checks['stage_weights'])
    assert changed['action']==original['action']
    assert changed['task']==original['task']
    assert changed['objective']['palm_position_weight']==original['objective']['palm_position_weight']
