"""Isolated rigid-object settling with the runtime's imported collision hulls.

No robot, controller, assistance or reference tracking force is constructed.
CPU MuJoCo results are diagnostics; use the guarded CUDA backend for matching
the robot runtime. Mesh appearance and object pose are never altered here.
"""
from pathlib import Path
import argparse,json,os


def assess_settling(summary, *, max_displacement_m=.05, max_rotation_deg=15.,
                    tail_displacement_m=.005, tail_rotation_deg=1.):
    """Report a bounded settling screen without treating CPU as GPU evidence.

    Early displacement and late drift are independent requirements. A robot
    can prop up an otherwise unstable object, so a successful robot rollout
    cannot replace this isolated check. Limits are caller-visible assumptions,
    not universal definitions of physically correct support.
    """
    import math
    limits=dict(max_displacement_m=max_displacement_m,max_rotation_deg=max_rotation_deg,
                tail_displacement_m=tail_displacement_m,tail_rotation_deg=tail_rotation_deg)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0
           for v in limits.values()):
        raise ValueError('Finite nonnegative settling limits required')
    tail=summary.get('settling_tail',{})
    values=dict(max_displacement_m=summary.get('max_displacement_m'),
                max_rotation_deg=summary.get('max_rotation_deg'),
                tail_displacement_m=tail.get('displacement_m'),tail_rotation_deg=tail.get('rotation_deg'))
    failed=[]
    if summary.get('finite') is not True:failed.append('nonfinite_states')
    for name,value in values.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            failed.append(name+':missing_or_invalid')
        elif value>limits[name]:failed.append(name+':above_limit')
    backend=summary.get('backend')
    return dict(schema='real2sim2real.settling-admission/1',passed=not failed,
        failed_checks=failed,measurements=values,limits=limits,backend=backend,
        matching_gpu_confirmation_required=backend!='MuJoCo-Warp CUDA',
        robot_present=summary.get('robot_present'),task_success_established=False,
        training_ready_established=False)


def settling_summary(states,*,recording_hz=50.,tail_seconds=1.):
    """Measure late drift separately from initial gravity/contact settling.

    Reports measurements only; callers own task-appropriate admission criteria.
    A small late drift does not certify the source reference or initial pose.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    q=np.asarray(states,float)
    if (q.ndim!=2 or q.shape[1]!=7 or len(q)<2 or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:,3:],axis=1),1.,atol=1e-4)
            or isinstance(recording_hz,bool) or not np.isfinite(recording_hz) or recording_hz<=0
            or isinstance(tail_seconds,bool) or not np.isfinite(tail_seconds) or tail_seconds<=0):
        raise ValueError('Finite rigid xyzw states, recording frequency and tail duration required')
    count=min(len(q),max(2,int(round(tail_seconds*recording_hz))+1));tail=q[-count:]
    rotation=Rotation.from_quat(tail[:,3:])
    angles=(rotation*rotation[0].inv()).magnitude()
    return dict(observed_tail_seconds=(count-1)/recording_hz,
        displacement_m=float(np.linalg.norm(tail[:,:3]-tail[0,:3],axis=1).max()),
        rotation_deg=float(np.rad2deg(angles).max()),
        max_linear_speed_mps=float(np.linalg.norm(np.diff(tail[:,:3],axis=0)*recording_hz,axis=1).max()),
        max_angular_speed_radps=float(((rotation[1:]*rotation[:-1].inv()).magnitude()*recording_hz).max()),
        initial_pose_quality_established=False,task_success_established=False)


def apply_scene_parameters(builder,parameters,*,collide_flag):
    """Explicit policy-matched materials for a single free rigid object.

    Opt-in only: legacy isolated probes keep their original defaults. Static
    supports add no object DOFs; reject another scene rather than changing its
    passive joints. This numerical helper has no GPU dependency.
    """
    import math
    if set(parameters)!={'contact_friction','object_free_joint_damping'}:
        raise ValueError('Explicit friction and free-object damping required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in parameters.values()):
        raise ValueError('Finite nonnegative scene parameters required')
    if builder.joint_dof_count!=6:raise ValueError('One free rigid object and static support required')
    passive=builder.custom_attributes.get('mujoco:dof_passive_damping')
    if passive is None:raise ValueError('MuJoCo passive damping attribute missing')
    for i in range(builder.shape_count):
        if int(builder.shape_flags[i])&collide_flag:builder.shape_material_mu[i]=float(parameters['contact_friction'])
    for i in range(6):passive.values[i]=float(parameters['object_free_joint_damping'])
    return dict(parameters)


def run(urdf,pose,output,*,seconds=3.,device='cpu',friction=1.,support_usda=None,
        support_mode='world_static',enable_multiccd=False,impratio=20.,scene_parameters=None,
        nconmax=4096,njmax=16384):
    import numpy as np
    urdf,output=Path(urdf),Path(output)
    if output.exists():raise FileExistsError('Preserve previous settling trials')
    pose=np.asarray(pose,float)
    if (pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5
        or not np.isfinite(seconds) or not 0<seconds<=30 or device not in ('cpu','cuda:0')
        or support_mode not in ('world_static','kinematic_legacy') or type(enable_multiccd) is not bool
        or isinstance(impratio,bool) or not np.isfinite(impratio) or impratio<=0
        or any(type(x) is not int or x<=0 for x in (nconmax,njmax))):
        raise ValueError('Rigid xyzw pose, finite ≤30-second duration and explicit device required')
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU settling diagnostics')
    # Reject invalid requests before loading the optional simulation backend.
    import newton,warp as wp
    from scipy.spatial.transform import Rotation
    from flash_chord.scene.objects import _decompose_object_collision
    from flash_chord.scene.collision import ShapeSpan
    from real2sim2real.artifacts import digest,write_json

    with wp.ScopedDevice(device):
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z,gravity=-9.81)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(str(urdf),floating=True,enable_self_collisions=False,collapse_fixed_joints=True,ignore_inertial_definitions=False)
        if builder.body_count!=1 or builder.joint_coord_count!=7:raise ValueError('One rigid free-root object required')
        builder.joint_q[:7]=pose.tolist();hulls=_decompose_object_collision(builder,ShapeSpan(0,builder.shape_count))
        support_binding=None
        if support_usda is not None:
            from flash_chord.scene.support import add_support_surfaces
            from .static_support import add_world_static_support
            support_usda=Path(support_usda)
            support_binding=(add_world_static_support(builder,support_usda) if support_mode=='world_static'
                             else add_support_surfaces(builder,support_usda))
        from dataclasses import replace
        builder.add_ground_plane(cfg=replace(builder.default_shape_cfg,mu=float(friction)))
        matched_scene=None
        if scene_parameters is not None:
            if support_mode!='world_static':raise ValueError('Policy-matched probe requires static isolated support')
            matched_scene=apply_scene_parameters(builder,scene_parameters,collide_flag=int(newton.ShapeFlags.COLLIDE_SHAPES))
        model=builder.finalize(device=device);a,b=model.state(),model.state();control=model.control()
        newton.eval_fk(model,model.joint_q,model.joint_qd,a)
        solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=device=='cpu',solver='newton',integrator='implicitfast',cone='pyramidal',impratio=impratio,iterations=100,ls_iterations=50,njmax=njmax,nconmax=nconmax,enable_multiccd=enable_multiccd)
        states=[a.body_q.numpy().copy()];steps=int(round(seconds*200))
        for step in range(steps):
            a.clear_forces();solver.step(a,b,control,None,.005);a,b=b,a
            if (step+1)%4==0:states.append(a.body_q.numpy().copy())
        all_states=np.asarray(states)
        support_motion=None
        if support_binding is not None and support_binding.body_id >= 0:
            support_states=all_states[:,support_binding.body_id]
            support_motion=dict(body_id=support_binding.body_id,
                **(settling_summary(support_states,tail_seconds=seconds)
                   if len(support_states)>=2 else
                   dict(available=False,reason='Probe is too short for two recorded states')))
        elif support_binding is not None:
            support_motion=dict(body_id=-1,representation='world_static_geometry',
                                fixed_by_construction=True,simulated_support_dofs=0)
        states=all_states[:,0];delta=np.linalg.norm(states[:,:3]-states[0,:3],axis=1)
        angles=Rotation.from_quat(states[:,3:])*Rotation.from_quat(states[0,3:]).inv();angle=np.rad2deg(angles.magnitude())
        value=dict(urdf_sha256=digest(urdf),backend='MuJoCo-C CPU' if device=='cpu' else 'MuJoCo-Warp CUDA',
            seconds=seconds,physics_hz=200.,recording_hz=50.,collision_hulls=hulls,
            body_mass_kg=model.body_mass.numpy().tolist(),body_com_m=model.body_com.numpy().tolist(),
            body_inertia_kg_m2=model.body_inertia.numpy().tolist(),max_displacement_m=float(delta.max()),
            max_rotation_deg=float(angle.max()),finite=bool(np.isfinite(states).all()),robot_present=False,object_assistance=False,
            stability_only=True,task_success_established=False)
        value.update(matched_scene_parameters=matched_scene,
            support_sha256=digest(support_usda) if support_usda is not None else None,
            support_shape_count=len(support_binding.shapes) if support_binding is not None else 0,
            support_scope='Explicit saved support geometry' if support_usda is not None else 'Ground plane only',
            support_motion=support_motion,support_mode=support_mode,
            simulated_body_count=model.body_count,simulated_joint_coordinate_count=model.joint_coord_count,
            physics_parameters=dict(timestep_s=.005,solver='newton',integrator='implicitfast',
                cone='pyramidal',impratio=impratio,iterations=100,ls_iterations=50,enable_multiccd=enable_multiccd,nconmax=nconmax,njmax=njmax))
        value['settling_tail']=(settling_summary(states) if len(states)>=2 else
            dict(available=False,reason='Probe is too short for two recorded states'))
        output.mkdir(parents=True);np.savez_compressed(output/'states.npz',body_q=states,all_body_q=all_states)
        write_json(output/'RESULTS.json',value);print(json.dumps(value,indent=2),flush=True)
        return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--urdf',required=True,type=Path);p.add_argument('--pose',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--support',type=Path);p.add_argument('--support-mode',choices=['world_static','kinematic_legacy'],default='world_static');p.add_argument('--impratio',type=float,default=20.,help='Match the target controller solver; recorded in results');p.add_argument('--enable-multiccd',action='store_true',help='Explicit multi-contact experiment; default solver behavior stays unchanged');a=p.parse_args()
    run(a.urdf,json.loads(a.pose.read_text())['pose_xyzw'],a.output,seconds=a.seconds,device=a.device,support_usda=a.support,support_mode=a.support_mode,enable_multiccd=a.enable_multiccd,impratio=a.impratio)


if __name__=='__main__':main()
