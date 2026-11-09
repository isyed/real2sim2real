from dataclasses import asdict,dataclass,replace
from hashlib import sha256
import json
from types import SimpleNamespace
import numpy as np
import pytest
jax = pytest.importorskip("jax", reason="Requires the optional actor-training runtime")
pytest.importorskip("newton", reason="Requires the optional simulation runtime")
import jax.numpy as jnp
from flash_chord.training.flash_sac.config import TrainingConfig,NetworkConfig
from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.training.flash_sac.checkpoint import save_actor_checkpoint
from real2sim2real.controller_actor_training.actor_warm_start import initialize,validate_metadata,training_context

@dataclass(frozen=True)
class State:
    def replace(self,**kwargs):return replace(self,**kwargs)

@dataclass(frozen=True)
class Network(State):
    params: object
    batch_stats: object
    optimizer_state: object

@dataclass(frozen=True)
class Learner(State):
    actor: object
    critic: object
    replay: object
    environment_steps: int=0
    global_update_step: int=0

def fixture(tmp_path):
    cfg=TrainingConfig(device='cpu',mixed_precision=False,network=NetworkConfig(actor_hidden_dim=16,actor_blocks=1))
    actor,saved=create_actor_template(cfg,8,2,device=jax.devices('cpu')[0])
    path=tmp_path/'actor.safetensors';save_actor_checkpoint(path,saved,metadata={
        'environment_steps':'123','policy_schema_json':json.dumps({'observation':{'dimension':8},'action':{'dimension':2}}),
        'resolved_config_json':json.dumps({'training':{'network':asdict(cfg.network)}})})
    _,fresh=create_actor_template(replace(cfg,seed=99),8,2,device=jax.devices('cpu')[0])
    replay=SimpleNamespace(size=0);critic=object();optimizer=object()
    learner=Learner(Network(fresh.params,fresh.batch_stats,optimizer),critic,replay)
    return cfg,actor,saved,path,learner

def test_actor_only_transfer_preserves_fresh_critic_optimizer_replay_and_actions(tmp_path):
    cfg,actor,saved,path,learner=fixture(tmp_path)
    transferred=initialize(learner,path,expected_sha256=sha256(path.read_bytes()).hexdigest())
    assert transferred.critic is learner.critic and transferred.replay is learner.replay
    assert transferred.actor.optimizer_state is learner.actor.optimizer_state
    assert transferred.environment_steps==transferred.global_update_step==0
    obs=jnp.ones((2,8))
    a=actor.apply({'params':saved.params,'batch_stats':saved.batch_stats},obs,training=False)
    b=actor.apply({'params':transferred.actor.params,'batch_stats':transferred.actor.batch_stats},obs,training=False)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)

@pytest.mark.parametrize('problem',['sha','steps','updates','replay'])
def test_reject_stale_or_nonfresh_transfer(tmp_path,problem):
    _,_,_,path,learner=fixture(tmp_path);digest=sha256(path.read_bytes()).hexdigest()
    if problem=='sha':digest='0'*64
    if problem=='steps':learner=replace(learner,environment_steps=1)
    if problem=='updates':learner=replace(learner,global_update_step=1)
    if problem=='replay':learner=replace(learner,replay=SimpleNamespace(size=1))
    with pytest.raises(ValueError):initialize(learner,path,expected_sha256=digest)

def test_metadata_rejects_resume_network_and_schema_changes():
    cfg=TrainingConfig();schema=json.dumps({'action':{'dimension':2},'observation':{'dimension':8}})
    network=dict(asdict(cfg.network),_target_='flash_chord.training.flash_sac.config.NetworkConfig')
    metadata=dict(policy_schema_json=schema,resolved_config_json=json.dumps({'training':{'network':network}}))
    validate_metadata(metadata,metadata,cfg)
    with pytest.raises(ValueError,match='mutually exclusive'):validate_metadata(metadata,metadata,replace(cfg,resume=True,checkpoint_path='x'))
    with pytest.raises(ValueError,match='schema'):validate_metadata(metadata,dict(policy_schema_json='{}'),cfg)
    with pytest.raises(ValueError,match='network'):validate_metadata(metadata,metadata,replace(cfg,network=replace(cfg.network,action_scale=3.)))

def test_context_requires_declared_new_task_and_restores_legacy_runner():
    from flash_chord.training.flash_sac import runner
    before=runner._run_training,runner.create_learner
    with pytest.raises(ValueError):
        with training_context('unused',expected_sha256='unused',evidence={}):pass
    with training_context('unused',expected_sha256='unused',evidence={'task_changed':True}):
        assert runner._run_training is not before[0]
    assert (runner._run_training,runner.create_learner)==before

def test_training_context_runs_transfer_and_saves_explicit_ancestry(tmp_path,monkeypatch):
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    cfg,_,_,path,learner=fixture(tmp_path);metadata=read_checkpoint_metadata(path)
    def fresh(*a,**kw):return learner,object()
    def run(env,config,**kwargs):
        state,_=runner.create_learner(config,None,None)
        proof=json.loads(kwargs['checkpoint_metadata']['actor_warm_start_json'])
        assert proof['parent_environment_steps']==123 and proof['exact_resume'] is False
        return state
    monkeypatch.setattr(runner,'create_learner',fresh);monkeypatch.setattr(runner,'_run_training',run)
    with training_context(path,expected_sha256=sha256(path.read_bytes()).hexdigest(),evidence={'task_changed':True}):
        result=runner._run_training(None,cfg,checkpoint_metadata=metadata,checkpoint_dir=tmp_path/'new')
    assert result.critic is learner.critic and result.actor.optimizer_state is learner.actor.optimizer_state
    assert json.loads((tmp_path/'new/actor_warm_start.json').read_text())['critic_optimizer_replay_counters_fresh']
    assert runner._run_training is run and runner.create_learner is fresh
