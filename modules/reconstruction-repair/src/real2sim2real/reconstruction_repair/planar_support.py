"""Observed-outline pose proposals for an explicitly reviewed resting phase.

The default model has two translations along a support and rotation around its
normal. An explicit optional mode permits a bounded tilt around the observed
seed while maintaining contact. Neither model permits unsupported lifting. A hypothesis
must independently pass native triangle, depth and temporal checks; no tracking
or reference acceptance is performed here.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def propose(vertices,base_rotation,seed_pose,intrinsics,object_mask,plane_camera,*,
            resting_evidence,max_plane_shift_m=.25,max_yaw_change_deg=30.,
            max_normal_correction_m=.15,clearance_m=.001,max_calls=150,
            allow_tilt=False,max_tilt_change_deg=10.):
    from real2sim2real.reconstruction.recovery_quality import rigid
    v,b,p,k,plane=map(lambda x:np.asarray(x,float),
        (vertices,base_rotation,seed_pose,intrinsics,plane_camera))
    mask=np.asarray(object_mask)
    limits=np.asarray([max_plane_shift_m,max_yaw_change_deg,max_normal_correction_m,clearance_m],float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
            or b.shape!=(3,3) or not np.isfinite(b).all() or not np.allclose(b@b.T,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(b),1,atol=1e-5) or not rigid(p)
            or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0
            or not np.allclose(k[2],[0,0,1]) or plane.shape!=(4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6) or plane[3]<=0
            or mask.ndim!=2 or mask.dtype!=bool or mask.sum()<30
            or not isinstance(resting_evidence,str) or not resting_evidence.strip()
            or not np.isfinite(limits).all() or np.any(limits[:3]<=0) or clearance_m<0
            or max_yaw_change_deg>60 or type(max_calls)!=int or max_calls<1
            or type(allow_tilt)!=bool or not np.isfinite(max_tilt_change_deg) or not 0<max_tilt_change_deg<=20):
        raise ValueError('Metric geometry, rigid seed/base, camera, mask and explicit bounded resting hypothesis required')
    n=plane[:3];axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);w=np.cross(n,u)
    center=(v.min(0)+v.max(0))/2;seed_center=p[:3,:3]@center+p[:3,3]
    anchor=p[:3,:3] if allow_tilt else b
    lowest=float(((v-center)@anchor.T@n).min())
    correction=clearance_m-(lowest+seed_center@n+plane[3])
    if abs(correction)>max_normal_correction_m:raise ValueError('Support correction exceeds the declared seed uncertainty')
    center0=seed_center+correction*n
    q=Rotation.from_matrix(p[:3,:3]@b.T).as_quat();dot=float(q[:3]@n)
    if np.hypot(dot,q[3])<1e-8:raise ValueError('Seed has no well-defined rotation around the support normal')
    yaw=2*np.arctan2(dot,q[3]);yaw=(yaw+np.pi)%(2*np.pi)-np.pi
    proxy=v[ConvexHull(v).vertices];yy,xx=np.nonzero(mask);observed=np.c_[xx,yy]
    angles=np.linspace(0,2*np.pi,32,endpoint=False);directions=np.c_[np.cos(angles),np.sin(angles)]
    target=np.max(observed@directions.T,axis=0);turn=np.deg2rad(max_yaw_change_deg)
    def pose(x):
        r=(Rotation.from_rotvec(x[2:]).as_matrix()@anchor if allow_tilt
            else Rotation.from_rotvec(n*(yaw+x[2])).as_matrix()@b)
        c=center0+x[0]*u+x[1]*w
        if allow_tilt:
            bottom=float(((v-center)@r.T@n).min())
            c=c+(clearance_m-(bottom+c@n+plane[3]))*n
        return r,c-r@center
    def residual(x):
        r,t=pose(x);camera=proxy@r.T+t
        if camera[:,2].min()<=.01:return np.full(32+len(x),1e6)
        pixels=camera@k.T;pixels=pixels[:,:2]/pixels[:,2:]
        error=np.max(pixels@directions.T,axis=0)-target
        return np.r_[error,x[:2]/max_plane_shift_m*.1,x[2:]/turn*(1. if allow_tilt else .1)]
    turn=np.deg2rad(max_tilt_change_deg)/np.sqrt(3) if allow_tilt else turn
    bounds=np.r_[np.repeat(max_plane_shift_m/np.sqrt(2),2),np.repeat(turn,3 if allow_tilt else 1)]
    fit=least_squares(residual,np.zeros(len(bounds)),bounds=(-bounds,bounds),
        loss='soft_l1',f_scale=2.,max_nfev=max_calls)
    r,t=pose(fit.x);camera=v@r.T+t
    if camera[:,2].min()<=.01:raise ValueError('Proposed support pose crosses the camera near plane')
    actual_normal=float((r@center+t-seed_center)@n)
    if abs(actual_normal)>max_normal_correction_m:raise ValueError('Fitted support correction exceeds the declared uncertainty')
    initial_r,initial_t=pose(np.zeros(len(bounds)))
    return r,t,dict(schema='real2sim2real.planar-support-proposal/1',resting_evidence=resting_evidence,
        proposal_only=True,model='bounded tilt and two support translations' if allow_tilt else 'two support translations and rotation about a fixed normal',
        normal_correction_m=actual_normal,plane_shift_m=float(np.linalg.norm(fit.x[:2])),
        yaw_change_deg=None if allow_tilt else float(np.rad2deg(fit.x[2])),
        tilt_change_deg=float(np.rad2deg(np.linalg.norm(fit.x[2:]))) if allow_tilt else None,
        minimum_support_height_m=float((camera@n+plane[3]).min()),
        initial_contact_rotation=initial_r.tolist(),initial_contact_translation=initial_t.tolist(),
        outline_rmse_before_px=float(np.sqrt(np.mean(residual(np.zeros(len(bounds)))[:32]**2))),
        outline_rmse_after_px=float(np.sqrt(np.mean(residual(fit.x)[:32]**2))),
        optimizer_calls=int(fit.nfev),mesh_changed=False,depth_used_for_fit=False,
        support_is_prediction=True,full_reference_approved=False,physics_validated=False,
        independent_native_depth_motion_checks_required=True)
