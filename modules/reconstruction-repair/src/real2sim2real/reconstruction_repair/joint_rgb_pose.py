"""Fixed-mesh multi-frame pose proposals using image outlines, not depth.

Only fit observations enter optimization. Query poses are inferred hypotheses;
withheld images and every declared original depth source must validate them.
The convex outline is a fitting proxy, never the final triangle-mask check.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Config:
    translation_bound_m: float = .15
    rotation_bound_deg: float = 25.
    pixel_sigma: float = 2.
    acceleration_sigma_m_s2: float = 20.
    angular_acceleration_sigma_deg_s2: float = 600.
    max_calls: int = 2400
    max_seconds: float = 120.
    restarts: int = 1


def fit(vertices, knot_times_s, initial_poses_camera, observations, query_times_s,
        *, locked, config=Config(), progress_callback=None, allow_unanchored=False):
    """Fit metric centers/rotations with fixed camera, mesh and explicit anchors.

    Each fit observation supplies K, unit image directions (Dx2), outline_px(D),
    outline_observed(D bool), support_plane_camera(4), source_provenance and
    ground_truth_used=False. Occluded outline directions are lower bounds only;
    they cannot pull a mesh boundary into the visible fragment. No depth or
    withheld observations are accepted by this interface. When no seed can be
    trusted, allow_unanchored=True explicitly permits their joint re-estimation;
    it does not approve the incoming/outgoing seams or infer a trusted anchor.
    """
    v,t,seeds,query=map(lambda x:np.asarray(x,float),
        (vertices,knot_times_s,initial_poses_camera,query_times_s))
    lock=np.asarray(locked);n=len(t);cfg=asdict(config)
    if (n<3 or t.shape!=(n,) or seeds.shape!=(n,4,4) or lock.shape!=(n,) or lock.dtype.kind!='b'
            or type(allow_unanchored) is not bool or (not lock.any() and not allow_unanchored)
            or lock.all() or v.ndim!=2 or v.shape[1]!=3 or len(v)<4
            or query.ndim!=1 or not len(query) or len(observations)!=n
            or not all(np.isfinite(x).all() for x in (v,t,seeds,query))
            or np.any(np.diff(t)<=0) or np.any(np.diff(query)<=0)
            or query[0]<t[0] or query[-1]>t[-1]
            or not np.allclose(seeds[:,3],[0,0,0,1])
            or not np.allclose(seeds[:,:3,:3]@seeds[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-5)
            or not np.allclose(np.linalg.det(seeds[:,:3,:3]),1,atol=1e-5)
            or any(isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in cfg.values())
            or type(config.max_calls) is not int or config.rotation_bound_deg>=90
            or type(config.restarts) is not int or not 1<=config.restarts<=7
            or progress_callback is not None and not callable(progress_callback)):
        raise ValueError('Finite metric geometry, rigid anchors, bounded query times and positive limits required')
    prepared=[]
    keys={'K','directions','outline_px','outline_observed','support_plane_camera','source_provenance','ground_truth_used'}
    for item in observations:
        if set(item)!=keys:raise ValueError('Only explicit fit-image observations are accepted; no depth or withheld data')
        k,axes,target,plane=[np.asarray(item[z],float) for z in ('K','directions','outline_px','support_plane_camera')]
        observed=np.asarray(item['outline_observed'])
        if (k.shape!=(3,3) or not np.allclose(k[2],[0,0,1]) or min(k[0,0],k[1,1])<=0
                or axes.ndim!=2 or axes.shape[1]!=2 or len(axes)<8 or target.shape!=(len(axes),)
                or not np.allclose(np.linalg.norm(axes,axis=1),1)
                or observed.shape!=target.shape or observed.dtype.kind!='b' or observed.sum()<8
                or plane.shape!=(4,) or not np.isclose(np.linalg.norm(plane[:3]),1)
                or not all(np.isfinite(x).all() for x in (k,axes,target,plane))
                or item['ground_truth_used'] is not False or not item['source_provenance']):
            raise ValueError('Calibrated image evidence, visible outline directions and source provenance required')
        prepared.append((k,axes,target,observed,plane))
    hull=v[ConvexHull(v).vertices];center=v.mean(0)
    seed_centers=np.einsum('nij,j->ni',seeds[:,:3,:3],center)+seeds[:,:3,3]
    editable=np.flatnonzero(~lock)
    limits=np.tile([config.translation_bound_m/np.sqrt(3)]*3+
        [np.deg2rad(config.rotation_bound_deg)/np.sqrt(3)]*3,len(editable))
    started=time.monotonic();calls=0;best=None;cost_best=float('inf');last_log=started

    def decode(x):
        changes=np.zeros((n,6));changes[editable]=x.reshape(-1,6)
        r=(Rotation.from_rotvec(changes[:,3:])*Rotation.from_matrix(seeds[:,:3,:3])).as_matrix()
        c=seed_centers+changes[:,:3];trans=c-np.einsum('nij,j->ni',r,center)
        r[lock]=seeds[lock,:3,:3];trans[lock]=seeds[lock,:3,3]
        return r,trans,c

    class Budget(Exception):pass
    def residual(x):
        nonlocal calls,best,cost_best,last_log
        if calls>=config.max_calls or time.monotonic()-started>=config.max_seconds:raise Budget()
        calls+=1;r,trans,centers=decode(x);parts=[]
        for i,(k,axes,target,observed,plane) in enumerate(prepared):
            camera=hull@r[i].T+trans[i];proj=camera@k.T
            uv=proj[:,:2]/np.maximum(proj[:,2:],.01);outline=np.max(uv@axes.T,axis=0)
            delta=outline-target
            parts.append(np.where(observed,delta,np.minimum(delta,0))/config.pixel_sigma)
            parts.append(np.array([max(0.,-np.min(camera@plane[:3]+plane[3])-.02)/.002,
                                   max(0.,.01-camera[:,2].min())/.001]))
        dt=np.diff(t);mid=(dt[1:]+dt[:-1])/2
        parts.append((np.diff(np.diff(centers,axis=0)/dt[:,None],axis=0)/mid[:,None]/config.acceleration_sigma_m_s2).ravel())
        omega=(Rotation.from_matrix(r[1:])*Rotation.from_matrix(r[:-1]).inv()).as_rotvec()/dt[:,None]
        parts.append((np.diff(omega,axis=0)/mid[:,None]/np.deg2rad(config.angular_acceleration_sigma_deg_s2)).ravel())
        parts.append(.1*x/limits)
        values=np.concatenate(parts);cost=float(np.sum(2*(np.sqrt(1+values*values)-1)))
        if cost<cost_best:best=x.copy();cost_best=cost
        if progress_callback and time.monotonic()-last_log>=15:
            progress_callback(dict(calls=calls,seconds=time.monotonic()-started,proxy_cost=cost_best));last_log=time.monotonic()
        return values

    zero=np.zeros(len(editable)*6);residual(zero);initial_cost=cost_best;budget=False
    starts=[zero];optimizers=[]
    # Finite orientation restarts share one global budget and remain relative to
    # the original seeds. Never compound bounds by treating a fitted pose as a
    # new anchor. Selection uses fit observations only.
    for axis in range(3):
        for sign in [-1.,1.]:
            start=zero.reshape(-1,6).copy();start[:,3+axis]=sign*np.deg2rad(config.rotation_bound_deg)/(2*np.sqrt(3))
            starts.append(start.ravel())
    for start in starts[:config.restarts]:
        try:
            solved=least_squares(residual,start,bounds=(-limits,limits),loss='soft_l1',max_nfev=config.max_calls,
                                 ftol=1e-6,xtol=1e-6,gtol=1e-6)
            optimizers.append(dict(status=int(solved.status),message=str(solved.message),nfev=int(solved.nfev)))
        except Budget:
            budget=True;break
    r,trans,centers=decode(best)
    poses=np.tile(np.eye(4),(len(query),1,1));poses[:,:3,:3]=Slerp(t,Rotation.from_matrix(r))(query).as_matrix()
    positions=np.stack([np.interp(query,t,centers[:,j]) for j in range(3)],axis=1)
    poses[:,:3,3]=positions-np.einsum('nij,j->ni',poses[:,:3,:3],center)
    for i in np.flatnonzero(lock):poses[query==t[i]]=seeds[i]
    shift=np.linalg.norm(centers-seed_centers,axis=1)
    angle=np.rad2deg((Rotation.from_matrix(r)*Rotation.from_matrix(seeds[:,:3,:3]).inv()).magnitude())
    return poses,dict(schema='real2sim2real.joint-rgb-pose/1',config=cfg,calls=calls,
        seconds=time.monotonic()-started,budget_reached=budget,initial_proxy_cost=initial_cost,best_proxy_cost=cost_best,
        completed_restarts=len(optimizers),optimizers=optimizers,restart_bounds_relative_to_original_seeds=True,
        fit_times_s=t.tolist(),query_times_s=query.tolist(),locked_knots=np.flatnonzero(lock).tolist(),
        unanchored_fit=bool(not lock.any()),external_seams_validated=False,
        max_knot_center_shift_m=float(shift.max()),max_knot_rotation_deg=float(angle.max()),
        source_provenance=[x['source_provenance'] for x in observations],depth_used_for_fitting=False,
        mesh_changed=False,camera_changed=False,heldout_observations_used=False,between_knot_poses_inferred=True,
        convex_outline_is_fitting_proxy=True,reference_approved=False,
        native_triangle_depth_support_motion_validation_required=True)
