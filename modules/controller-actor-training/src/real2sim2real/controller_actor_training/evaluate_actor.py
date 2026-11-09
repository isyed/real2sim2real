"""Record full unassisted residual-actor physics and separate kinematic targets.

Loads only an actor checkpoint for inference; no optimizer, critic or replay.
"""
from pathlib import Path
import argparse
import json
import os

from real2sim2real.artifacts import digest, write_json
from .frozen import materialize


def evaluate(policy,checkpoint,output,*,record_reference=True,controller_override=None):
    policy,checkpoint,output=map(lambda p:Path(p).resolve(),[policy,checkpoint,output])
    if output.exists():
        raise ValueError('Preserve previous rollouts; use a new output')
    manifest,config=materialize(policy)
    os.environ['R2S2R_ASSETS']=str(policy/'assets')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import numpy as np
    from flash_chord.evaluation import harness
    from flash_chord.data import load_reference
    native=load_reference(policy/manifest.get('native_reference',manifest['reference']))
    output.mkdir(parents=True)
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    saved_metadata=read_checkpoint_metadata(checkpoint)
    saved_config=json.loads(saved_metadata['resolved_config_json'])
    if (Path(saved_config['task']['parquet']).resolve()!=Path(config['task']['parquet']).resolve()
            or saved_config['action']!=config['action']
            or saved_config['scene']['support']!=config['scene']['support']):
        raise ValueError('Actor reference/controller differs from the portable evaluation bundle')
    override_record=None
    effective_metadata=saved_metadata
    if controller_override is not None:
        from .controller_override import apply_reference_pull
        override_path=Path(controller_override).resolve()
        override_values=json.loads(override_path.read_text())
        effective_config=apply_reference_pull(saved_config,override_values)
        effective_metadata=dict(saved_metadata,resolved_config_json=json.dumps(effective_config))
        override_record=dict(path=str(override_path),sha256=digest(override_path),values=override_values,
            inference_only=True,checkpoint_tensors_changed=False,training_performed=False)
    from .evaluation_build import provenance
    def record(env,jax_env,policy_action,actor_state,key,observation,**kwargs):
        bodies,joints,velocities,actions,contacts=[],[],[],[],[]
        step=jax_env.step
        def observed(action):
            result=step(action)
            bodies.append(env.state_0.body_q.numpy().copy())
            joints.append(env.state_0.joint_q.numpy().copy())
            velocities.append(env.state_0.joint_qd.numpy().copy())
            actions.append(np.asarray(jax.device_get(action)).copy())
            if len(bodies)==1 or len(bodies)%100==0:
                print(f'evaluation progress: {len(bodies)}/{kwargs["step_count"]} control frames',flush=True)
            if env.contact is not None:
                contacts.append(env.contact.raw_contact_count.numpy().copy())
            return result
        jax_env.step=observed
        try:
            rollout=harness.record_object_pose_rollout(env,jax_env,policy_action,actor_state,key,observation,**kwargs)
        finally:
            jax_env.step=step
        correction=np.asarray(actions)
        if not len(correction) or not np.isfinite(correction).all():
            raise ValueError('Nonfinite or missing actor corrections')
        np.savez_compressed(output/'physical_states.npz',body_q=np.asarray(bodies),
            joint_q=np.asarray(joints),joint_qd=np.asarray(velocities),
            correction_actions=correction,contact_counts=np.asarray(contacts),
            contact_slot_hand_ids=env.contact.slot_hand_ids.numpy() if env.contact is not None else np.array([]))
        audit=dict(actor_loaded=True,actor_trained=True,training_steps=int(saved_metadata['environment_steps']),
            checkpoint_sha256=digest(checkpoint),
            correction_max_abs=float(np.max(np.abs(correction))),object_assistance_scale=0.,
            hand_closing_enabled=getattr(env.action,'hand_controller',None) is not None,
            body_labels=list(env.scene.model.body_label),
            pelvis_body_id=int(env.observation_strategy.pelvis_body_id),
            object_body_id=int(env.observation_strategy.object_body_id),
            source_frames=native.num_frames,control_frames=kwargs['step_count'],control_fps=float(env.config.sim.fps),
            source_fps=float(manifest.get('source_fps',0.)),policy_sha256=digest(policy/'manifest.json'),
            reference_sha256=digest(config['task']['parquet']),reset_mode='explicit',start_frame=0)
        if override_record is not None:audit['controller_override']=override_record
        if 'deployment_relocation_json' in saved_metadata:
            audit['deployment_relocation']=json.loads(saved_metadata['deployment_relocation_json'])
        if not audit['hand_closing_enabled']:
            raise ValueError('Fixed hand controller was not instantiated')
        if record_reference:
            target=[]
            for frame in range(kwargs['step_count']):
                env.reset(frame_id=frame)
                target.append(env.state_0.body_q.numpy().copy())
            np.savez_compressed(output/'kinematic_states.npz',body_q=np.asarray(target))
            audit['kinematic_states']='Separate reset-to-reference states, no physical execution'
        write_json(output/'audit.json',audit)
        return rollout
    outcome=harness.evaluate_checkpoint(checkpoint,world_count=1,start_frame=0,
        reset_mode='explicit',rollout_recorder=record,parquet_resolver=lambda c,r:config['task']['parquet'],
        provenance_builder=provenance,object_trajectories_output=output/'trajectory.parquet',
        eval_script=Path(__file__).with_name("evaluation_build.py"),
        **({'metadata_reader':lambda path:effective_metadata} if override_record is not None else {}))
    from .source_clock import export_source_clock
    export_source_clock(output/'trajectory.parquet', native)
    harness.write_evaluation_report(output/'metrics.json',outcome)
    print(json.dumps(harness.evaluation_scalars(outcome),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--no-reference-states',action='store_true')
    parser.add_argument('--controller-override',type=Path)
    args=parser.parse_args()
    evaluate(args.policy,args.checkpoint,args.output,record_reference=not args.no_reference_states,controller_override=args.controller_override)
