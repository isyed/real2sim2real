"""Unassisted physics and matched kinematic targets for a frozen controller.

No actor, optimizer, critic or replay buffer is constructed. Kinematic target
states are recorded separately after the physical rollout, never as its output.
"""
from pathlib import Path
import argparse
import json
import os

from real2sim2real.artifacts import digest, write_json
from .frozen import materialize


def zero_policy(checkpoint,training,observation,action_dim,*,deterministic,device):
    import jax
    import jax.numpy as jnp
    if action_dim!=43 or deterministic is not True:
        raise ValueError('Expected the deterministic G1/Dex3 residual interface')
    @jax.jit
    def act(state,obs,key):
        return jnp.zeros((obs.shape[0],action_dim),dtype=jnp.float32),key
    return None,act,jax.random.PRNGKey(42),{}


def evaluate(policy,output,*,record_reference=True):
    policy,output=Path(policy).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous rollouts; use a new output')
    manifest,config=materialize(policy)
    os.environ['R2S2R_ASSETS']=str(policy/'assets')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import numpy as np
    from flash_chord.evaluation import harness
    output.mkdir(parents=True)
    def metadata(path):
        if Path(path)!=policy/'manifest.json':
            raise ValueError('Unexpected frozen policy identity')
        return dict(resolved_config_json=json.dumps(config),environment_steps='0')
    def provenance(**kwargs):
        value=harness.evaluation_provenance(**kwargs)
        import hashlib
        value['eval_code_sha256']=hashlib.sha256((value['eval_code_sha256']+
            digest(Path(__file__))+digest(Path(__file__).with_name('frozen.py'))).encode()).hexdigest()
        return value
    def record(env,jax_env,policy_action,actor_state,key,observation,**kwargs):
        bodies,joints,velocities,actions,contacts=[],[],[],[],[]
        step=jax_env.step
        def observed(action):
            result=step(action)
            bodies.append(env.state_0.body_q.numpy().copy())
            joints.append(env.state_0.joint_q.numpy().copy())
            velocities.append(env.state_0.joint_qd.numpy().copy())
            actions.append(np.asarray(jax.device_get(action)).copy())
            if env.contact is not None:
                contacts.append(env.contact.raw_contact_count.numpy().copy())
            return result
        jax_env.step=observed
        try:
            rollout=harness.record_object_pose_rollout(env,jax_env,policy_action,actor_state,key,observation,**kwargs)
        finally:
            jax_env.step=step
        correction=np.asarray(actions)
        if not len(correction) or np.max(np.abs(correction))!=0:
            raise ValueError('Nonzero or missing correction actions in frozen rollout')
        np.savez_compressed(output/'physical_states.npz',body_q=np.asarray(bodies),
            joint_q=np.asarray(joints),joint_qd=np.asarray(velocities),
            correction_actions=correction,contact_counts=np.asarray(contacts),
            contact_slot_hand_ids=env.contact.slot_hand_ids.numpy() if env.contact is not None else np.array([]))
        audit=dict(actor_loaded=False,actor_trained=False,training_steps=0,
            correction_max_abs=0.,object_assistance_scale=0.,
            hand_closing_enabled=getattr(env.action,'hand_controller',None) is not None,
            body_labels=list(env.scene.model.body_label),
            pelvis_body_id=int(env.observation_strategy.pelvis_body_id),
            object_body_id=int(env.observation_strategy.object_body_id),
            source_frames=kwargs['step_count'],control_fps=float(env.config.sim.fps),
            source_fps=float(manifest.get('source_fps',0.)),policy_sha256=digest(policy/'manifest.json'),
            reference_sha256=digest(config['task']['parquet']),reset_mode='explicit',start_frame=0)
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
    outcome=harness.evaluate_checkpoint(policy/'manifest.json',world_count=1,start_frame=0,
        reset_mode='explicit',metadata_reader=metadata,policy_loader=zero_policy,
        rollout_recorder=record,parquet_resolver=lambda c,r:config['task']['parquet'],
        provenance_builder=provenance,object_trajectories_output=output/'trajectory.parquet',
        eval_script=Path(__file__))
    harness.write_evaluation_report(output/'metrics.json',outcome)
    print(json.dumps(harness.evaluation_scalars(outcome),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-reference-states',action='store_true')
    args=parser.parse_args()
    evaluate(args.policy,args.output,record_reference=not args.no_reference_states)
