"""Build fresh FlashSAC experiments from a portable frozen-controller bundle.

Only the reference, assets and controller settings are reused. Actor, critic,
optimizer and replay start fresh unless matching resume files are explicit.
"""
from copy import deepcopy
from pathlib import Path


def configuration(policy, output, *, environment_steps, worlds=256,
                  orientation_weight=0.1, resume_state=None, resume_replay=None,
                  assistance_thresholds=None):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from .frozen import materialize
    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError('worlds must be a positive integer')
    if isinstance(environment_steps, bool) or not isinstance(environment_steps, int) or environment_steps < worlds:
        raise ValueError('Set a positive explicit transition budget covering one collection')
    if not 0 <= orientation_weight <= 1:
        raise ValueError('orientation_weight must be between zero and one')
    if bool(resume_state) != bool(resume_replay):
        raise ValueError('Exact continuation requires matching learner state and replay')
    _, frozen = materialize(policy)
    with initialize_config_module(version_base='1.3', config_module='flash_chord.configs'):
        cfg = compose(config_name='train', overrides=['experiment=g1_bounded_flash_sac',
            f'scene.world_count={worlds}', 'training.network.action_scale=2.0',
            '+objective.object_position_tight_weight=1.0', '+objective.object_position_tight_std=0.05',
            f'+objective.object_orientation_weight={orientation_weight}', '+objective.object_orientation_std=0.75',
            *[f'+curriculum.stages.{i}.objective_weights.object_orientation={orientation_weight}' for i in range(4)]])
    c = OmegaConf.to_container(cfg, resolve=True)
    if assistance_thresholds is not None:
        values=list(assistance_thresholds)
        if (len(values)!=len(c['curriculum']['stages'])
                or any(type(v) is not int or v<worlds for v in values)
                or any(a>=b for a,b in zip(values,values[1:]))):
            raise ValueError('One increasing positive transition threshold per curriculum stage required')
        rounded=[v//worlds*worlds for v in values]
        if any(a>=b for a,b in zip(rounded,rounded[1:])):
            raise ValueError('Curriculum thresholds collapse after collection rounding')
        c['curriculum']['thresholds']=rounded
    c['task'] = deepcopy(frozen['task'])
    c['action'] = deepcopy(frozen['action'])
    c['env']['action'] = deepcopy(c['action'])
    c['scene']['support'] = frozen['scene']['support']
    c['training']['total_environment_steps'] = environment_steps // worlds * worlds
    c['training']['checkpoint'].update(save_interval_environment_steps=5000000 // worlds * worlds,
                                     save_replay=True, load_replay_path=str(resume_replay) if resume_replay else None)
    # Keep the learning-rate schedule fixed across bounded continuations.
    c['training']['optimization']['learning_rate_schedule_updates'] = 234372
    c['training'].update(resume=bool(resume_state), checkpoint_path=str(resume_state) if resume_state else None)
    c['training']['curriculum'] = deepcopy(c['curriculum'])
    c['output_dir'] = str(Path(output).resolve())
    c['logging']['mode'] = 'disabled'
    return c


def validate(c):
    from flash_chord.configuration import instantiate_typed
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.flash_sac.config import TrainingConfig
    from omegaconf import OmegaConf
    env = instantiate_typed(OmegaConf.create(c['env']), RLEnvConfig)
    training = instantiate_typed(OmegaConf.create(c['training']), TrainingConfig)
    for name in ('action', 'observation', 'objective', 'reset', 'termination', 'sim'):
        if c[name] != c['env'][name]:
            raise ValueError('Runtime configuration mismatch: '+name)
    if c['training']['curriculum'] != c['curriculum']:
        raise ValueError('Runtime curriculum mismatch')
    terms = env.objective.term_names
    weights = [dict(zip(terms, s.weights_for(terms))) for s in training.curriculum.stages]
    if not c['action'].get('hand_closing'):
        raise ValueError('Fixed hand controller is required')
    return dict(objective_terms=list(terms), stage_weights=weights, assistance_scales=[s.voc_scale for s in training.curriculum.stages],
                assistance_thresholds=list(training.curriculum.thresholds),
                environment_steps=training.total_environment_steps, worlds=training.world_count,
                replay_capacity=training.replay.capacity, resume=training.resume)
