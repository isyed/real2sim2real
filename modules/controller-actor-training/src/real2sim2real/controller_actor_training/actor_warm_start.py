"""Actor-only transfer into a fresh FlashSAC learner for a declared new task.

Unlike resume, this preserves only actor parameters and batch statistics. The
critic, optimizers, replay, temperature, counters and random state stay fresh.
Heavy dependencies are imported only when the operation is requested.
"""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path


def validate_metadata(metadata, current_metadata, configuration):
    from dataclasses import asdict
    if configuration.resume:
        raise ValueError('Actor warm-start and exact learner resume are mutually exclusive')
    if json.loads(metadata['policy_schema_json']) != json.loads(current_metadata['policy_schema_json']):
        raise ValueError('Actor action/observation schema differs')
    previous = json.loads(metadata['resolved_config_json'])
    network = {k:v for k,v in previous['training']['network'].items() if k != '_target_'}
    if network != asdict(configuration.network):
        raise ValueError('Actor network or action distribution configuration differs')
    return previous


def initialize(learner, checkpoint, *, expected_sha256):
    import jax
    import numpy as np
    from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint
    path = Path(checkpoint)
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Warm-start checkpoint changed')
    if int(learner.environment_steps) != 0 or int(learner.global_update_step) != 0 or int(learner.replay.size) != 0:
        raise ValueError('Warm-start requires a fresh learner and empty replay')
    template = ActorInferenceState(params=learner.actor.params, batch_stats=learner.actor.batch_stats)
    state, _ = load_actor_checkpoint(path, template)
    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state)):
        raise ValueError('Nonfinite actor parameters')
    return learner.replace(actor=learner.actor.replace(params=state.params, batch_stats=state.batch_stats))


@contextmanager
def training_context(checkpoint, *, expected_sha256, evidence):
    """Opt-in bridge; all legacy training behavior is restored on exit.

    The runner creates its ordinary fresh learner before loading the actor.
    Callers must declare the task/reference change in checkpoint ancestry.
    """
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    if not evidence or not evidence.get('task_changed'):
        raise ValueError('Explicit new-task transfer evidence is required')
    original_run, original_create = runner._run_training, runner.create_learner
    def run(env, config, **kwargs):
        metadata = read_checkpoint_metadata(checkpoint)
        current = kwargs.get('checkpoint_metadata')
        if current is None:
            raise ValueError('Current checkpoint schema metadata is required')
        validate_metadata(metadata, current, config)
        ancestry = dict(schema='real2sim2real.actor-warm-start/1', actor_sha256=expected_sha256,
            parent_environment_steps=int(metadata['environment_steps']),
            exact_resume=False, actor_parameters_and_batch_statistics_only=True,
            critic_optimizer_replay_counters_fresh=True, evidence=evidence)
        kwargs['checkpoint_metadata'] = dict(current, actor_warm_start_json=json.dumps(ancestry, sort_keys=True))
        def create(*args, **create_kwargs):
            learner, executables = original_create(*args, **create_kwargs)
            learner = initialize(learner, checkpoint, expected_sha256=expected_sha256)
            if kwargs.get('checkpoint_dir'):
                target = Path(kwargs['checkpoint_dir']) / 'actor_warm_start.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(ancestry, indent=2)+'\n')
            return learner, executables
        runner.create_learner = create
        try:
            return original_run(env, config, **kwargs)
        finally:
            runner.create_learner = original_create
    runner._run_training = run
    try:
        yield
    finally:
        runner._run_training, runner.create_learner = original_run, original_create
