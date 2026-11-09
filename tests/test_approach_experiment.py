import json
from pathlib import Path

import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.approach_experiment import ApproachExperiment


def protocol(tmp_path, improve):
    run=tmp_path/'experiment';policy=run/'policy';policy.mkdir(parents=True)
    (policy/'manifest.json').write_text('{}')
    old=run/'existing5m_window.safetensors';old.write_bytes(b'unchanged existing actor')
    (run/'prepared.json').write_text(json.dumps(dict(policy_sha256=digest(policy/'manifest.json'),incumbent_sha256=digest(old))))
    experiment=ApproachExperiment(run,tmp_path/'source',tmp_path/'original5m')
    calls=[];initializations=[]
    def config(target,previous):
        initializations.append(previous)
        if previous is not None:
            assert Path(previous).is_relative_to(run/'training')
            assert Path(previous)!=old
        return {'training':{'total_environment_steps':target}}, {'validated':True}
    def execute(args,label,seconds):
        calls.append(label);target=int(label.split('_')[1]);folder=run/'training';folder.mkdir(exist_ok=True)
        for name in ('policy','state','replay'):(folder/f'{name}_{target}.safetensors').write_bytes((name+str(target)).encode())
    def evaluate(label,actor=None):
        survival=4.
        if label.isdigit():survival=8. if improve else 3.
        return dict(valid=True,survival_s=survival,min_survival_s=survival,
            upright_intended_contact_s=0.,upright_palm_rmse_first4s_m=.2,auc=.3)
    experiment.configuration=config;experiment.execute=execute;experiment.evaluate=evaluate
    return experiment,calls,initializations,old


def test_regression_stops_at5m_and_preserves_retained_actor(tmp_path):
    experiment,calls,initializations,old=protocol(tmp_path,False)
    before=old.read_bytes();result=experiment.train_and_evaluate()
    assert calls==['train_5000000']
    assert initializations==[None]
    assert old.read_bytes()==before
    assert result['preferred_short_window_controller']=='existing5m'
    assert result['full_task_deployment_approved'] is False


def test_continuation_uses_only_new_matching_learner_not_existing_comparison(tmp_path):
    experiment,calls,initializations,old=protocol(tmp_path,True)
    result=experiment.train_and_evaluate()
    assert calls==['train_5000000','train_10000000']
    assert initializations==[None,experiment.run/'training/policy_5000000.safetensors']
    assert result['preferred_fresh_checkpoint'] in ('5000000','10000000')
    assert old.read_bytes()==b'unchanged existing actor'


def test_changed_saved_learner_is_not_silently_reused(tmp_path):
    experiment,calls,initializations,old=protocol(tmp_path,False)
    experiment.train_and_evaluate()
    (experiment.run/'training/state_5000000.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        experiment.train_and_evaluate()


def test_missing_replay_cannot_resume_a_new_profile(tmp_path):
    experiment=ApproachExperiment(tmp_path,tmp_path/'source',tmp_path/'old')
    with pytest.raises(ValueError,match='learner state/replay'):
        experiment.configuration(10000000,tmp_path/'training/policy_4999936.safetensors')
