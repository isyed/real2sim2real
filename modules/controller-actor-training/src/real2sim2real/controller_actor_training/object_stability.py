"""Isolated rigid-object settling with the runtime's imported collision hulls.

No robot, controller, assistance or reference tracking force is constructed.
CPU MuJoCo results are diagnostics; use the guarded CUDA backend for matching
the robot runtime. Mesh appearance and object pose are never altered here.
"""
from pathlib import Path
import argparse,json,os


def run(urdf,pose,output,*,seconds=3.,device='cpu',friction=1.):
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
        from dataclasses import replace
        builder.add_ground_plane(cfg=replace(builder.default_shape_cfg,mu=float(friction)))
        model=builder.finalize(device=device);a,b=model.state(),model.state();control=model.control()
        newton.eval_fk(model,model.joint_q,model.joint_qd,a)
        solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=device=='cpu',solver='newton',integrator='implicitfast',cone='pyramidal',impratio=20.,iterations=100,ls_iterations=50,njmax=16384,nconmax=4096)
        states=[a.body_q.numpy().copy()];steps=int(round(seconds*200))
        for step in range(steps):
            a.clear_forces();solver.step(a,b,control,None,.005);a,b=b,a
            if (step+1)%4==0:states.append(a.body_q.numpy().copy())
        states=np.asarray(states)[:,0];delta=np.linalg.norm(states[:,:3]-states[0,:3],axis=1)
        angles=Rotation.from_quat(states[:,3:])*Rotation.from_quat(states[0,3:]).inv();angle=np.rad2deg(angles.magnitude())
        value=dict(urdf_sha256=digest(urdf),backend='MuJoCo-C CPU' if device=='cpu' else 'MuJoCo-Warp CUDA',
            seconds=seconds,physics_hz=200.,recording_hz=50.,collision_hulls=hulls,
            body_mass_kg=model.body_mass.numpy().tolist(),body_com_m=model.body_com.numpy().tolist(),
            body_inertia_kg_m2=model.body_inertia.numpy().tolist(),max_displacement_m=float(delta.max()),
            max_rotation_deg=float(angle.max()),finite=bool(np.isfinite(states).all()),robot_present=False,object_assistance=False,
            stability_only=True,task_success_established=False)
        output.mkdir(parents=True);np.savez_compressed(output/'states.npz',body_q=states)
        write_json(output/'RESULTS.json',value);print(json.dumps(value,indent=2),flush=True)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--urdf',required=True,type=Path);p.add_argument('--pose',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');a=p.parse_args()
    run(a.urdf,json.loads(a.pose.read_text())['pose_xyzw'],a.output,seconds=a.seconds,device=a.device)
