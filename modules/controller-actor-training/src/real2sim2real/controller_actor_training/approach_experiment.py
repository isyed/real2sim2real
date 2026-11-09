"""Reusable bounded approach experiment against an unchanged existing actor.

The host supplies guarded process execution and progress callbacks. This module
does not change hardware limits, provision cloud resources or select a submission.
Heavy dependencies load only within their stage.
"""
import copy
import json
from pathlib import Path
import shutil


class ApproachExperiment:
    def __init__(self, run, source_policy, comparison_actor, *, stop_native=241,
                 targets=(5000000, 10000000), execute=None, progress=None):
        self.run, self.source, self.old_actor = map(Path, (run, source_policy, comparison_actor))
        self.policy = self.run / 'policy'
        if type(stop_native) is not int or stop_native < 2:
            raise ValueError('Explicit native window required')
        if tuple(targets) != (5000000, 10000000):
            raise ValueError('This bounded protocol supports5M and conditional10M')
        self.stop_native, self.targets = stop_native, tuple(targets)
        self.execute = execute
        self.progress = progress or (lambda *args, **kwargs: None)

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def write(path, value):
        from real2sim2real.artifacts import write_json
        write_json(path, value)

    def configuration(self, target, previous=None):
        from . import residual, unassisted_pilot, approach_profile, policy_physics
        state = replay = prior = None
        if previous:
            previous = Path(previous)
            step = previous.stem.split('_')[-1]
            state = previous.with_name('state_' + step + '.safetensors')
            replay = previous.with_name('replay_' + step + '.safetensors')
            if not state.is_file() or not replay.is_file():
                raise ValueError('Matching learner state/replay required')
            from flash_chord.training.checkpoint import read_checkpoint_metadata
            prior = json.loads(read_checkpoint_metadata(state)['resolved_config_json'])
        cfg = residual.configuration(self.policy, self.run / 'training', environment_steps=target,
                                     resume_state=state, resume_replay=replay)
        frozen = self.read(self.policy / 'manifest.json')['config']
        cfg, contact = policy_physics.match_contact_mode(cfg, frozen)
        cfg, physics = policy_physics.match_scene(cfg, frozen)
        for key in ('support', 'decompose_objects', 'object_scale_min', 'object_scale_max', 'object_scale_seed'):
            cfg['scene'][key] = frozen['scene'][key]
        cfg, mode = unassisted_pilot.configure(cfg, replay_capacity=131072, frame_zero_probability=1.)
        cfg, profile = approach_profile.configure(cfg, prior_configuration=prior)
        check = residual.validate(cfg)
        if any(check['assistance_scales']) or cfg['voc_scale_override'] != 0:
            raise ValueError('Every stage must be unassisted')
        return cfg, dict(validation=check, contact_mode=contact, physics=physics, mode=mode, profile=profile)

    @staticmethod
    def actor_forward(path):
        import numpy as np
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf
        from flash_chord.training.checkpoint import read_checkpoint_metadata
        from flash_chord.training.flash_sac.config import TrainingConfig
        from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
        from flash_chord.configuration import instantiate_typed
        from real2sim2real.artifacts import digest
        metadata = read_checkpoint_metadata(path)
        cfg = json.loads(metadata['resolved_config_json'])
        schema = json.loads(metadata['policy_schema_json'])
        training = instantiate_typed(OmegaConf.create(cfg['training']), TrainingConfig)
        obs = jnp.zeros((1, schema['observation']['dimension']), jnp.float32)
        state, act, key, _ = load_policy_for_inference(str(path), training, obs,
            schema['action']['dimension'], deterministic=True, device=jax.devices('cpu')[0])
        result, _ = act(state, obs, key)
        if result.shape != (1, schema['action']['dimension']) or not np.isfinite(np.asarray(result)).all():
            raise ValueError('Nonfinite or incompatible comparison actor')
        return dict(passed=True, cpu_only=True, action_shape=list(result.shape), actor_sha256=digest(path))

    def prepare(self):
        import numpy as np
        import pyarrow.parquet as pq
        from . import reference_window, frozen, verify_frozen
        from .relocate_checkpoint import relocate
        from flash_chord.training.checkpoint import read_checkpoint_metadata
        from real2sim2real.artifacts import load, digest
        if (self.run / 'prepared.json').exists():
            raise FileExistsError('Preserve prepared experiment')
        self.run.mkdir(parents=True, exist_ok=True)
        original = self.read(self.source / 'manifest.json')
        load(self.source, kinds={'frozen-controller'})
        metadata = read_checkpoint_metadata(self.old_actor)
        old = json.loads(metadata['resolved_config_json'])
        old_steps = int(metadata['environment_steps'])
        if not 4999000 <= old_steps <= 5000000:
            raise ValueError('This comparison requires the declared retained5M actor')
        if not (self.run / 'reference/artifact.json').exists():
            reference_window.prepare(self.source / 'reference', self.run / 'reference',
                                     start=0, stop=self.stop_native, hardlink=True)
        if not (self.policy / 'artifact.json').exists():
            frozen.package(self.run / 'reference', self.source / 'assets', self.policy,
                           hardlink=True, collision_cache=self.source / 'assets/.cache/convex')
            manifest = self.read(self.policy / 'manifest.json')
            for key in ('scene', 'sim', 'action', 'collision'):
                manifest['config'][key] = copy.deepcopy(original['config'][key])
            for key in ('action', 'sim'):
                manifest['config']['env'][key] = copy.deepcopy(manifest['config'][key])
            self.write(self.policy / 'manifest.json', manifest)
            artifact = self.read(self.policy / 'artifact.json')
            artifact['files']['manifest']['sha256'] = digest(self.policy / 'manifest.json')
            self.write(self.policy / 'artifact.json', artifact)
        forwarded = verify_frozen.verify(self.policy, expected_frames=self.stop_native)
        self.write(self.run / 'model_preflight.json', forwarded)
        row = pq.read_table(self.policy / 'reference/reference.parquet').to_pylist()[0]
        parent = pq.read_table(self.source / original['native_reference']).to_pylist()[0]
        for key in ('robot_root_position', 'robot_root_wxyz', 'robot_joint_positions', 'ee_pose_w',
                    'object_body_position', 'object_body_wxyz'):
            if not np.array_equal(np.asarray(row[key]), np.asarray(parent[key])[:self.stop_native]):
                raise ValueError('Window changed native pose values: ' + key)
        cfg, _ = self.configuration(self.targets[0])
        replacements = {}
        for field, value in [('task.parquet', cfg['task']['parquet']), ('reference.parquet', cfg['task']['parquet']),
                             ('action.policy_dir', cfg['action']['policy_dir']), ('env.action.policy_dir', cfg['action']['policy_dir'])]:
            node = old
            for part in field.split('.'):
                node = node[part]
            replacements[field] = (node, value)
        incumbent = self.run / 'existing5m_window.safetensors'
        if not incumbent.exists():
            transfer = relocate(self.old_actor, incumbent, replacements, allow_deployment_chain=True,
                evidence=dict(reference_horizon_changed=True, source_frames=len(parent['robot_root_position']),
                    target_frames=self.stop_native, native_prefix_arrays_unchanged=True, actor_tensors_unchanged=True,
                    normalized_phase_and_lookahead_change=True, inference_only=True, exact_training_continuation=False))
            self.write(self.run / 'existing_actor_transfer.json', transfer)
        self.write(self.run / 'actor_preflight.json', self.actor_forward(incumbent))
        for target in self.targets:
            _, proof = self.configuration(target)
            self.write(self.run / f'configuration_{target}.json', proof)
        receipt = dict(source_frames=self.stop_native, control_frames=forwarded['control_frames'],
            source_window=[0, self.stop_native], seconds=(self.stop_native-1)/float(row['fps']),
            source_policy_sha256=digest(self.source / 'manifest.json'), policy_sha256=digest(self.policy / 'manifest.json'),
            incumbent_sha256=digest(incumbent), incumbent_training_steps=old_steps,
            fresh_actor_critic_optimizer_replay=True, zero_object_assistance=True, all_starts_frame_zero=True,
            replay_capacity=131072, worlds=256, checkpoints=list(self.targets), maximum_steps=self.targets[-1],
            trial_count=3, full_task_deployment_approved=False, automatic_submission_replacement=False,
            intended_contact_frames_native=[int(np.asarray(v, bool).sum()) for v in row['hand_contact_active']])
        self.write(self.run / 'prepared.json', receipt)
        return receipt

    def evaluate(self, label, actor=None):
        import numpy as np
        import pyarrow.parquet as pq
        from .approach_diagnostics import summarize
        row = pq.read_table(self.policy / self.read(self.policy / 'manifest.json')['reference']).to_pylist()[0]
        if list(row['hand_sides']) != ['left', 'right']:
            raise ValueError('Explicit two-hand order required')
        phase = np.asarray(row['hand_contact_active'], bool).T
        trials = []
        for trial in (1, 2, 3):
            out = self.run / f'evaluations/{label}/trial_{trial}'
            if not (out / 'audit.json').exists():
                if out.exists():
                    raise ValueError('Preserve incomplete trial before recovery')
                self.progress('unassisted_evaluation', checkpoint=label, trial=trial)
                args = ['-m', 'real2sim2real.controller_actor_training.' + ('evaluate_actor' if actor else 'evaluate_frozen'),
                        '--policy', self.policy, '--output', out]
                if actor:
                    args += ['--checkpoint', actor]
                self.execute(args, 'eval_' + label + '_' + str(trial), 1200)
            audit, metrics = self.read(out / 'audit.json'), self.read(out / 'metrics.json')
            if (audit['object_assistance_scale'] != 0 or metrics['non_finite_worlds'] or audit['start_frame'] != 0
                    or audit['control_frames'] != len(phase)):
                raise ValueError('Complete unassisted frame-zero trial required')
            with np.load(out / 'physical_states.npz') as p, np.load(out / 'kinematic_states.npz') as k:
                result = summarize(p['body_q'], k['body_q'], p['contact_counts'], p['contact_slot_hand_ids'], phase,
                    pelvis_body=audit['pelvis_body_id'], palm_bodies=[audit['body_labels'].index('g1/' + s + '_hand_palm_link')
                    for s in ('left', 'right')], fps=audit['control_fps'])
            result.update(auc=metrics['metrics']['add_auc'], mppe_cm=metrics['metrics']['mppe_cm'])
            trials.append(result)
        fields = ('survival_s', 'upright_intended_contact_s', 'upright_palm_rmse_first4s_m', 'auc', 'mppe_cm')
        valid = all(t[name] is not None and np.isfinite(t[name]) for t in trials for name in fields)
        means = {name: float(np.mean([t[name] for t in trials])) if all(t[name] is not None for t in trials) else None for name in fields}
        return dict(valid=bool(valid), trials=trials, min_survival_s=min(t['survival_s'] for t in trials),
                    **means, scope='Own reconstructed window; not full-task or competition scores')

    @staticmethod
    def gate(metrics, label):
        from .approach_diagnostics import continuation
        return dict(comparator='existing5m', **continuation(metrics['existing5m'], metrics[label], frozen_auc=metrics['baseline']['auc']))

    def train_and_evaluate(self):
        from omegaconf import OmegaConf
        from real2sim2real.artifacts import digest
        prepared = self.read(self.run / 'prepared.json')
        if digest(self.policy / 'manifest.json') != prepared['policy_sha256']:
            raise ValueError('Prepared policy changed')
        if digest(self.run / 'existing5m_window.safetensors') != prepared['incumbent_sha256']:
            raise ValueError('Retained comparison actor changed')
        metrics = self.read(self.run / 'metrics.json') if (self.run / 'metrics.json').exists() else {}
        for label, actor in [('baseline', None), ('existing5m', self.run / 'existing5m_window.safetensors')]:
            if label not in metrics:
                metrics[label] = self.evaluate(label, actor)
                self.write(self.run / 'metrics.json', metrics)
        actor = None
        for target in self.targets:
            if shutil.disk_usage(self.run).free < 3 * 1024**3:
                raise RuntimeError('Preserve checkpoints: less than3GiB free')
            cfg, proof = self.configuration(target, actor)
            folder = self.run / f'configuration_{target}'
            folder.mkdir(exist_ok=True)
            OmegaConf.save(OmegaConf.create(cfg), folder / 'train.yaml')
            step = cfg['training']['total_environment_steps']
            current = self.run / f'training/policy_{step}.safetensors'
            marker = self.run / f'checkpoint_{target}.json'
            if not marker.exists():
                if current.exists():
                    raise ValueError('Recover and verify existing checkpoint before retry')
                self.progress('fresh_actor_training' if actor is None else 'continuing_same_unassisted_learner', target_steps=target)
                self.execute(['-m', 'real2sim2real.controller_actor_training.train', '--config-path', folder,
                              '--config-name', 'train'], f'train_{target}', 6000)
                files = {name: self.run / f'training/{name}_{step}.safetensors' for name in ('policy', 'state', 'replay')}
                if not all(p.is_file() for p in files.values()):
                    raise ValueError('Incomplete learner/state/replay set')
                self.write(marker, dict(steps=step, files={k: dict(path=str(p), sha256=digest(p)) for k, p in files.items()}, configuration=proof))
            else:
                for item in self.read(marker)['files'].values():
                    if digest(item['path']) != item['sha256']:
                        raise ValueError('Retained learner checkpoint changed')
            actor = current
            label = str(target)
            if label not in metrics:
                metrics[label] = self.evaluate(label, actor)
                self.write(self.run / 'metrics.json', metrics)
            decision = self.gate(metrics, label)
            self.write(self.run / f'gate_{target}.json', decision)
            if target == self.targets[0] and not decision['continue_to_10m']:
                break
        qualified = [k for k in metrics if k.isdigit() and self.gate(metrics, k)['continue_to_10m']]
        preferred = max(qualified, key=lambda k: (metrics[k]['survival_s'], metrics[k]['upright_intended_contact_s'],
                        -metrics[k]['upright_palm_rmse_first4s_m'])) if qualified else None
        result = dict(metrics=metrics, preferred_fresh_checkpoint=preferred,
            preferred_short_window_controller=preferred or 'existing5m', fresh_training_performed=True,
            object_assistance=0., source_window=[0, self.stop_native], full_task_deployment_approved=False)
        self.write(self.run / 'FINAL_RESULTS.json', result)
        return result
