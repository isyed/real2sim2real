"""CPU collision screen at explicit recorded body poses, without integration.

Uses the policy's imported collision geometry and filters. Newton's CPU narrow
phase is a diagnostic; it is not a MuJoCo contact-force or stability test.
"""
import os
from pathlib import Path


def summarize(body_poses, body_labels, shape_bodies, owners, contacts):
    import numpy as np
    from scipy.spatial.transform import Rotation
    q=np.asarray(body_poses,float);bodies=np.asarray(shape_bodies,int)
    if (q.shape!=(len(body_labels),7) or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:,3:],axis=1),1.,atol=1e-4)
            or bodies.ndim!=1 or len(owners)!=len(bodies)
            or np.any(bodies < -1) or np.any(bodies>=len(q))):
        raise ValueError('Finite rigid body poses and complete shape ownership required')
    rotations=Rotation.from_quat(q[:,3:]).as_matrix();rows=[]
    def world(point,body):
        return np.asarray(point,float) if body<0 else rotations[body]@point+q[body,:3]
    for c in contacts:
        i,j=int(c['shape0']),int(c['shape1'])
        if min(i,j)<0 or max(i,j)>=len(bodies):raise ValueError('Contact shape out of range')
        n=np.asarray(c['normal'],float);p0=np.asarray(c['point0'],float);p1=np.asarray(c['point1'],float)
        margins=np.asarray([c['margin0'],c['margin1']],float)
        if (n.shape!=(3,) or p0.shape!=(3,) or p1.shape!=(3,) or
                not np.isfinite(np.r_[n,p0,p1,margins]).all() or
                not np.isclose(np.linalg.norm(n),1.,atol=1e-4) or np.any(margins<0)):
            raise ValueError('Finite body-frame contact points, unit normal and nonnegative margins required')
        p0w,p1w=world(p0,bodies[i]),world(p1,bodies[j]);raw=float((p1w-p0w)@n)
        gap=raw-float(margins.sum())
        rows.append(dict(shape0=i,shape1=j,owner0=owners[i],owner1=owners[j],
            body0=body_labels[bodies[i]] if bodies[i]>=0 else 'world',
            body1=body_labels[bodies[j]] if bodies[j]>=0 else 'world',
            point0_world_m=p0w.tolist(),point1_world_m=p1w.tolist(),
            raw_signed_gap_m=raw,effective_signed_gap_m=gap,
            penetration_m=max(0.,-gap),margin_sum_m=float(margins.sum())))
    groups={}
    for row in rows:
        pair='/'.join(sorted([row['owner0'],row['owner1']]))
        group=groups.setdefault(pair,dict(contact_count=0,maximum_penetration_m=0.))
        group['contact_count']+=1
        group['maximum_penetration_m']=max(group['maximum_penetration_m'],row['penetration_m'])
    return dict(groups=groups,contacts=sorted(rows,key=lambda r:r['penetration_m'],reverse=True),
        contact_force_measured=False,physical_integration=False,
        limitation='Only generated near-contact pairs are screened; absence is not a global clearance proof.')


def audit(policy,body_poses,body_labels,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU collision screening')
    policy,output=Path(policy).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior clearance results')
    os.environ['R2S2R_ASSETS']=str(policy/'assets')
    import numpy as np
    import warp as wp
    from omegaconf import OmegaConf
    from real2sim2real.artifacts import digest,write_json
    from .frozen import materialize
    from flash_chord.assets import ASSETS_DIR
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    manifest,c=materialize(policy);cfg=OmegaConf.create(c)
    if Path(ASSETS_DIR).resolve()!=(policy/'assets').resolve():
        raise ValueError('Policy assets were imported for another root; screen in a fresh process')
    if cfg.scene.world_count!=1:raise ValueError('Screen one policy world at a time')
    with wp.ScopedDevice('cpu'):
        scene=setup_scene(parquet=cfg.task.parquet,control_fps=cfg.sim.fps,motion_speed=cfg.task.motion_speed,
            embodiment=instantiate_typed(cfg.embodiment,Embodiment),
            collision=instantiate_typed(cfg.collision,CollisionPolicy),world_count=1,
            include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,object_free_joint_damping=cfg.scene.object_free_joint_damping,
            source_frame_playback=cfg.task.source_frame_playback,
            object_scale_min=cfg.scene.get('object_scale_min',1.),object_scale_max=cfg.scene.get('object_scale_max',1.),
            object_scale_seed=cfg.scene.get('object_scale_seed',0)).scene
        model=scene.model
        if list(model.body_label)!=list(body_labels):raise ValueError('Recorded body layout differs from policy')
        q=np.asarray(body_poses,dtype=np.float32)
        if (q.shape!=(model.body_count,7) or not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q[:,3:],axis=1),1.,atol=1e-4)):
            raise ValueError('Finite rigid body poses matching the model required')
        state=model.state();state.body_q.assign(q);contact=model.collide(state)
        count=int(contact.rigid_contact_count.numpy()[0])
        arrays={k:getattr(contact,'rigid_contact_'+k).numpy() for k in ['shape0','shape1','point0','point1','normal','margin0','margin1']}
        if count>len(arrays['shape0']):raise ValueError('Collision screen overflow; no clearance conclusion')
        owners=['ground']*model.shape_count
        for key in ['robot','objects','support']:
            for i in getattr(scene.collision_layout,key).ids():owners[i]=key
        result=summarize(q,body_labels,model.shape_body.numpy(),owners,
                         [{k:v[i] for k,v in arrays.items()} for i in range(count)])
        output.mkdir(parents=True)
        np.savez_compressed(output/'contacts.npz',body_q=q,shape_bodies=model.shape_body.numpy(),
            **{k:v[:count] for k,v in arrays.items()})
        result.update(policy_sha256=digest(policy/'manifest.json'),backend='Newton CPU narrow phase',
            object_shapes=len(scene.collision_layout.objects),support_shapes=len(scene.collision_layout.support),
            body_labels=list(body_labels),contact_record_sha256=digest(output/'contacts.npz'))
        write_json(output/'RESULTS.json',result);return result


def replay_solver_options(controller_sim):
    """Resolve omitted optional settings with the runtime's own defaults."""
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.runtime.sim import SimConfig
    effective=instantiate_typed(OmegaConf.create(controller_sim),SimConfig)
    return {key:getattr(effective,key) for key in ['solver','integrator','cone','impratio','iterations','ls_iterations','nconmax','njmax','enable_multiccd','use_mujoco_contacts']}


def mujoco_replay(policy, rollout, output, *, frame_stride=10):
    """CPU MuJoCo contact check at saved GPU states, without stepping physics.

    This cross-check is useful when Newton mesh witnesses report implausibly
    large gaps. It measures CPU collision distances, not GPU contact forces.
    Refuse mismatched kinematics rather than silently replay a different pose.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or type(frame_stride) is not int or frame_stride<=0:
        raise ValueError('Explicit CPU-only execution and positive frame stride required')
    import json
    import numpy as np
    import newton,warp as wp,mujoco
    from omegaconf import OmegaConf
    from real2sim2real.artifacts import digest,write_json
    from .frozen import materialize
    policy,rollout,output=map(lambda p:Path(p).resolve(),[policy,rollout,output])
    if output.exists():raise FileExistsError('Preserve contact checks')
    os.environ['R2S2R_ASSETS']=str(policy/'assets')
    from flash_chord.assets import ASSETS_DIR
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    if Path(ASSETS_DIR).resolve()!=(policy/'assets').resolve():raise ValueError('Use a fresh policy asset process')
    manifest,c=materialize(policy);cfg=OmegaConf.create(c)
    receipt=json.loads((rollout/'audit.json').read_text())
    if receipt['policy_sha256']!=digest(policy/'manifest.json') or cfg.scene.world_count!=1:
        raise ValueError('One world and an exact policy-bound rollout required')
    with np.load(rollout/'physical_states.npz') as z:
        body=z['body_q'].copy();q=z['joint_q'].copy();qd=z['joint_qd'].copy()
    if not all(np.isfinite(v).all() for v in [body,q,qd]):raise ValueError('Finite saved states required')
    with wp.ScopedDevice('cpu'):
        scene=setup_scene(parquet=cfg.task.parquet,control_fps=cfg.sim.fps,motion_speed=cfg.task.motion_speed,
            embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),
            world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,object_free_joint_damping=cfg.scene.object_free_joint_damping,
            source_frame_playback=cfg.task.source_frame_playback,
            object_scale_min=cfg.scene.get('object_scale_min',1.),object_scale_max=cfg.scene.get('object_scale_max',1.),
            object_scale_seed=cfg.scene.get('object_scale_seed',0)).scene
        model=scene.model
        if list(model.body_label)!=receipt['body_labels'] or body.shape[1:]!=(model.body_count,7):
            raise ValueError('Saved/model body layout differs')
        if q.shape!=(len(body),model.joint_coord_count) or qd.shape!=(len(body),model.joint_dof_count):
            raise ValueError('Saved/model joint layout differs')
        options=replay_solver_options(cfg.sim)
        if not options['use_mujoco_contacts']:raise ValueError('Native MuJoCo-contact policy required')
        solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,**options)
        state=model.state();m,d=solver.mj_model,solver.mj_data
        mapping=solver.mjc_body_to_newton.numpy()[0];rows=[];maximum_position_error=0.
        def label(geom):
            b=int(mapping[int(m.geom_bodyid[geom])]);return 'world' if b<0 else model.body_label[b]
        def owner(name):return 'robot' if name.startswith('g1/') else 'support' if name=='support' else 'ground' if name=='world' else 'objects'
        for frame in sorted(set(range(0,len(body),frame_stride))|{len(body)-1}):
            state.body_q.assign(body[frame]);state.joint_q.assign(q[frame]);state.joint_qd.assign(qd[frame])
            solver._update_mjc_data(d,model,state)
            # No mj_step: preserve the observed GPU motion exactly.
            mujoco.mj_forward(m,d)
            valid=np.flatnonzero(mapping>=0)
            error=float(np.max(np.abs(d.xpos[valid]-body[frame,mapping[valid],:3])))
            if error>1e-4:raise ValueError('CPU replay kinematics differ from saved GPU poses')
            maximum_position_error=max(maximum_position_error,error);groups={};contacts=[]
            for contact in d.contact[:d.ncon]:
                names=[label(int(g)) for g in contact.geom];pair='/'.join(sorted(owner(n) for n in names))
                penetration=max(0.,-float(contact.dist));group=groups.setdefault(pair,dict(count=0,maximum_penetration_m=0.))
                group['count']+=1;group['maximum_penetration_m']=max(group['maximum_penetration_m'],penetration)
                if pair=='robot/support':contacts.append(dict(bodies=names,distance_m=float(contact.dist),position_world_m=contact.pos.tolist()))
            rows.append(dict(control_frame=frame,time_s=(frame+1)/cfg.sim.fps,groups=groups,robot_support_contacts=contacts))
        result=dict(backend='MuJoCo-C CPU collision replay; no integration',policy_sha256=digest(policy/'manifest.json'),
            rollout_states_sha256=digest(rollout/'physical_states.npz'),frame_stride=frame_stride,samples=rows,
            maximum_replayed_position_error_m=maximum_position_error,contact_forces_measured=False,
            full_continuous_clearance_established=False,physics_options=options,
            mujoco_version=mujoco.__version__,newton_version=getattr(newton,'__version__','unknown'))
        output.mkdir(parents=True);write_json(output/'RESULTS.json',result);return result
