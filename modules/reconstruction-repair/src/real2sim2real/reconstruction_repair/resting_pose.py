"""Bounded upright/resting-pose proposal from explicitly reviewed geometry.

The caller supplies the object's upright axis, support plane and local pivot.
No object category, contact phase, camera evidence or approval is inferred.
The result still requires image/depth, continuity and CPU/GPU settling gates.
"""
def propose(vertices, pose_xyzw, axis_world, support_plane_world, pivot_local, *,
            fraction=1., clearance_m=.001, tangent_offset_m=(0.,0.),
            max_turn_deg=25., max_translation_m=.15):
    import numpy as np
    from scipy.spatial.transform import Rotation
    v=np.asarray(vertices,float);p=np.asarray(pose_xyzw,float);axis=np.asarray(axis_world,float)
    plane=np.asarray(support_plane_world,float);pivot=np.asarray(pivot_local,float);offset=np.asarray(tangent_offset_m,float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or p.shape!=(7,) or axis.shape!=(3,)
            or plane.shape!=(4,) or pivot.shape!=(3,) or offset.shape!=(2,)
            or not all(np.isfinite(x).all() for x in [v,p,axis,plane,pivot,offset])
            or not np.isclose(np.linalg.norm(p[3:]),1.,atol=1e-6)
            or not np.isclose(np.linalg.norm(axis),1.,atol=1e-6)
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6)):
        raise ValueError('Finite metric vertices, rigid xyzw pose, unit axis/plane and explicit pivot required')
    values=np.asarray([fraction,clearance_m,max_turn_deg,max_translation_m],float)
    if (not np.isfinite(values).all() or not 0<=fraction<=1 or clearance_m<0
            or not 0<max_turn_deg<180 or max_translation_m<=0):
        raise ValueError('Bounded fraction, nonnegative clearance and positive correction limits required')
    n=plane[:3];turn,_=Rotation.align_vectors(n[None],axis[None])
    actual=turn.as_rotvec()*fraction;angle=float(np.rad2deg(np.linalg.norm(actual)))
    if angle>max_turn_deg+1e-9:raise ValueError('Reviewed-axis rotation exceeds declared bound')
    old=Rotation.from_quat(p[3:]).as_matrix();new=Rotation.from_rotvec(actual).as_matrix()@old
    pivot_world=pivot@old.T+p[:3];position=pivot_world-pivot@new.T
    lift=clearance_m-float(((v@new.T+position)@n+plane[3]).min());position+=n*lift
    x=np.cross(n,[0.,1.,0.])
    if np.linalg.norm(x)<1e-8:x=np.cross(n,[1.,0.,0.])
    x/=np.linalg.norm(x);y=np.cross(n,x);position+=x*offset[0]+y*offset[1]
    delta=float(np.linalg.norm(position-p[:3]))
    if delta>max_translation_m+1e-9:raise ValueError('Resting-pose translation exceeds declared bound')
    result=np.r_[position,Rotation.from_matrix(new).as_quat()]
    return result,dict(schema='real2sim2real.resting-pose-proposal/1',rotation_change_deg=angle,
        translation_change_m=delta,normal_clearance_adjustment_m=lift,clearance_m=clearance_m,
        tangent_offset_m=offset.tolist(),upright_axis_explicitly_supplied=True,
        geometry_changed=False,inertia_changed=False,physics_validated=False,
        source_image_validation_required=True,proposal_only=True)
