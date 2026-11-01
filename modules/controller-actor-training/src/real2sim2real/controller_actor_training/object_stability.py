"""Isolated rigid-object settling with the runtime's imported collision hulls.

No robot, controller, assistance or reference tracking force is constructed.
CPU MuJoCo results are diagnostics; use the guarded CUDA backend for matching
the robot runtime. Mesh appearance and object pose are never altered here.
"""
from pathlib import Path
import argparse,json,os


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


def run(urdf,pose,output,*,seconds=3.,device='cpu',friction=1.,support_usda=None):
    import numpy as np
    import newton,warp as wp
    from scipy.spatial.transform import Rotation
    from flash_chord.scene.objects import _decompose_object_collision
    from flash_chord.scene.collision import ShapeSpan
    from real2sim2real.artifacts import digest,write_json
    urdf,output=Path(urdf),Path(output)
    if output.exists():raise FileExistsError('Preserve previous settling trials')
    pose=np.asarray(pose,float)
    if (pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5
        or not np.isfinite(seconds) or not 0<seconds<=30 or device not in ('cpu','cuda:0')):
        raise ValueError('Rigid xyzw pose, finite ≤30-second duration and explicit device required')
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU settling diagnostics')
    with wp.ScopedDevice(device):
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z,gravity=-9.81)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(str(urdf),floating=True,enable_self_collisions=False,collapse_fixed_joints=True,ignore_inertial_definitions=False)
        if builder.body_count!=1 or builder.joint_coord_count!=7:raise ValueError('One rigid free-root object required')
        builder.joint_q[:7]=pose.tolist();hulls=_decompose_object_collision(builder,ShapeSpan(0,builder.shape_count))
        support_binding=None
        if support_usda is not None:
            from flash_chord.scene.support import add_support_surfaces
            support_usda=Path(support_usda)
            support_binding=add_support_surfaces(builder,support_usda)
        from dataclasses import replace
        builder.add_ground_plane(cfg=replace(builder.default_shape_cfg,mu=float(friction)))
        model=builder.finalize(device=device);a,b=model.state(),model.state();control=model.control()
        newton.eval_fk(model,model.joint_q,model.joint_qd,a)
        solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=device=='cpu',solver='newton',integrator='implicitfast',cone='pyramidal',impratio=20.,iterations=100,ls_iterations=50,njmax=16384,nconmax=4096)
        states=[a.body_q.numpy().copy()];steps=int(round(seconds*200))
        for step in range(steps):
            a.clear_forces();solver.step(a,b,control,None,.005);a,b=b,a
            if (step+1)%4==0:states.append(a.body_q.numpy().copy())
        all_states=np.asarray(states)
        support_motion=None
        if support_binding is not None:
            support_states=all_states[:,support_binding.body_id]
            support_motion=dict(body_id=support_binding.body_id,
                **(settling_summary(support_states,tail_seconds=seconds)
                   if len(support_states)>=2 else
                   dict(available=False,reason='Probe is too short for two recorded states')))
        states=all_states[:,0];delta=np.linalg.norm(states[:,:3]-states[0,:3],axis=1)
        angles=Rotation.from_quat(states[:,3:])*Rotation.from_quat(states[0,3:]).inv();angle=np.rad2deg(angles.magnitude())
        value=dict(urdf_sha256=digest(urdf),backend='MuJoCo-C CPU' if device=='cpu' else 'MuJoCo-Warp CUDA',
            seconds=seconds,physics_hz=200.,recording_hz=50.,collision_hulls=hulls,
            body_mass_kg=model.body_mass.numpy().tolist(),body_com_m=model.body_com.numpy().tolist(),
            body_inertia_kg_m2=model.body_inertia.numpy().tolist(),max_displacement_m=float(delta.max()),
            max_rotation_deg=float(angle.max()),finite=bool(np.isfinite(states).all()),robot_present=False,object_assistance=False,
            stability_only=True,task_success_established=False)
        value.update(support_sha256=digest(support_usda) if support_usda is not None else None,
            support_shape_count=len(support_binding.shapes) if support_binding is not None else 0,
            support_scope='Explicit saved support geometry' if support_usda is not None else 'Ground plane only',
            support_motion=support_motion)
        value['settling_tail']=(settling_summary(states) if len(states)>=2 else
            dict(available=False,reason='Probe is too short for two recorded states'))
        output.mkdir(parents=True);np.savez_compressed(output/'states.npz',body_q=states,all_body_q=all_states)
        write_json(output/'RESULTS.json',value);print(json.dumps(value,indent=2),flush=True)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--urdf',required=True,type=Path);p.add_argument('--pose',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--support',type=Path);a=p.parse_args()
    run(a.urdf,json.loads(a.pose.read_text())['pose_xyzw'],a.output,seconds=a.seconds,device=a.device,support_usda=a.support)
