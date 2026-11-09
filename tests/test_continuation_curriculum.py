from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training.continuation_curriculum import amend_reset_curriculum, continuation_review
from flash_chord.lifecycle.curriculum import FixedCurriculum, CurriculumStage


def config():
    return dict(scene=dict(world_count=256),objective=dict(object_orientation_weight=.1),
        reset=dict(reset_freeze_steps=50,reset_voc_scale=1),
        curriculum=dict(thresholds=[4,7,10,30],stages=[dict(voc_scale=0.,objective_weights=dict(object_position=1),immediate_first_frame_probability=.5)]),
        training=dict(resume=True,optimization=dict(rate=.00015),curriculum={}))


def test_absolute_boundary_and_no_assistance_restart_or_reward_change():
    old=config();original=deepcopy(old)
    new=amend_reset_curriculum(old,boundaries=[40000000,60000000],probabilities=[.5,.75])
    stages=[CurriculumStage(**s) for s in new['curriculum']['stages']]
    schedule=FixedCurriculum(tuple(new['curriculum']['thresholds']),tuple(stages))
    for step,p in [(20000000,.5),(24999936,.5),(39999744,.5),(40000000,.75),(60000000,.75)]:
        assert schedule.stage_at(step).immediate_first_frame_probability==p
        assert schedule.stage_at(step).voc_scale==0
    assert old==original and new['objective']==old['objective'] and new['reset']==old['reset']
    assert new['training']['curriculum']==new['curriculum']
    assert new['training']['optimization']==old['training']['optimization']


@pytest.mark.parametrize('ends,ps', [([4,5],[.5,.75]),([40000000],[.5,.75]),([60000000,40000000],[.5,.75]),([40000000,60000000],[.75,.5]),([40000000],[float('nan')])])
def test_bad_schedule_rejected(ends,ps):
    with pytest.raises(ValueError):amend_reset_curriculum(config(),boundaries=ends,probabilities=ps)


def metric(auc=.3,mppe=10,**extra):
    return dict(auc=auc,mppe_cm=mppe,first_low_pelvis_frame=1000,valid=True,sp_sr=0,mp_sr=0,**extra)


def test_single_dip_does_not_stop_but_two_severe_regressions_do():
    b=metric();bad=metric(.2)
    assert continuation_review(b,None,bad)['continue_training']
    assert not continuation_review(b,bad,bad)['continue_training']


def test_major_gate_progress_and_success_exception():
    b=metric()
    assert not continuation_review(b,b,b,major_review=True)['continue_training']
    assert continuation_review(b,b,metric(.31),major_review=True)['continue_training']
    m=metric(.29);m['sp_sr']=1/3
    assert continuation_review(b,b,m,major_review=True)['continue_training']
    m['auc']=.2
    assert not continuation_review(b,b,m,major_review=True)['continue_training']


def test_invalid_execution_stops_immediately():
    m=metric();m['valid']=False
    assert not continuation_review(metric(),None,m)['continue_training']
