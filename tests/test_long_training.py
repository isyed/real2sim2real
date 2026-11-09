from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training.long_training import configuration,review


def test_fresh_recipe_keeps_policy_physics_and_bounds_all_assistance(monkeypatch):
    from real2sim2real.controller_actor_training import frozen,residual
    policy=dict(scene=dict(world_count=1,support=True,contact_friction=.8,object_free_joint_damping=.01),
        env=dict(sim=dict(nconmax=512,njmax=1024,impratio=1.,enable_multiccd=True)),collision=dict(robot_support_collision=True))
    base=dict(scene=dict(world_count=256),sim={},env=dict(sim={}),collision={},
        curriculum=dict(stages=[dict(voc_scale=0.,objective_weights=dict(object_orientation=.1))]),
        reset=dict(reset_freeze_steps=50,reset_voc_scale=1.),training=dict(curriculum={},resume=False))
    original=deepcopy(base)
    monkeypatch.setattr(frozen,'materialize',lambda p:({},policy))
    monkeypatch.setattr(residual,'configuration',lambda *args,**kwargs:deepcopy(base))
    c=configuration('policy','output',environment_steps=5_000_000)
    assert base==original and c['scene']['world_count']==256
    assert c['sim']==policy['env']['sim']==c['env']['sim']
    assert c['collision']==policy['collision']
    assert c['reset']['reset_freeze_steps']==50
    assert c['reset']['reset_voc_scale']==0 and c['env']['reset']==c['reset']
    assert [s['voc_scale'] for s in c['curriculum']['stages']]==[1.,.5,.1,0.,0.]
    assert [s['immediate_first_frame_probability'] for s in c['curriculum']['stages']]==[.5,.5,.5,.5,.75]
    from flash_chord.lifecycle.curriculum import FixedCurriculum,CurriculumStage
    schedule=FixedCurriculum(tuple(c['curriculum']['thresholds']),tuple(CurriculumStage(**s) for s in c['curriculum']['stages']))
    for step,scale,probability in [(0,1.,.5),(4_000_000,.5,.5),(7_000_000,.1,.5),(9_999_872,0.,.5),(10_000_000,0.,.5),(39_999_744,0.,.5),(40_000_000,0.,.75),(60_000_000,0.,.75)]:
        assert schedule.stage_at(step).voc_scale==scale
        assert schedule.stage_at(step).immediate_first_frame_probability==probability
    assert c['training']['curriculum']==c['curriculum']


def metric(auc=.3,valid=True):
    return dict(auc=auc,valid=valid,mppe_cm=10.,first_low_pelvis_frame=800.,sp_sr=0.,mp_sr=0.)


def test_no_early_performance_gate_but_invalid_execution_stops():
    for target in [5_000_000,10_000_000]:
        assert review(metric(),metric(.1),metric(.1),target=target)['continue_training']
        assert not review(metric(),None,metric(valid=False),target=target)['continue_training']
    assert not review(metric(),metric(.1),metric(.1),target=15_000_000)['continue_training']


def test_major_40m_gate_and_bounded_targets():
    assert not review(metric(),metric(),metric(),target=40_000_000)['continue_training']
    assert review(metric(),metric(),metric(.31),target=40_000_000)['continue_training']
    with pytest.raises(ValueError):configuration('policy','output',environment_steps=65_000_000)
    with pytest.raises(ValueError):review(metric(),None,metric(),target=25_000_000)
