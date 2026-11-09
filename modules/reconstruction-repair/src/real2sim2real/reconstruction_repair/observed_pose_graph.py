"""Sparse source-observation pose fitting with explicit temporal regularization.

Contour extrema and fixed first-surface depth correspondences are fitting
proxies. The caller must independently validate original triangles, visibility,
native depth, support and motion. This optimizer never approves a reference.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FitConfig:
    max_seconds: float=600.
    max_calls: int=4000
    pixel_sigma: float=2.
    depth_sigma_m: float=.03
    support_sigma_m: float=.002
    support_penetration_m: float=.015
    translation_component_bound_m: float=.40
    rotation_component_bound_deg: float=30.
    acceleration_sigma_m_s2: float=9.
    angular_acceleration_sigma_deg_s2: float=900.
    proxy_vertices: int=512


def fit(vertices,reference,initial,intrinsics,timestamps_s,outline_directions,
        outline_targets_px,outline_weights,depth_points_local,depth_target_m,
        depth_weights,support_planes,locked,*,config=FitConfig(),progress_callback=None):
    arrays=[np.asarray(x,float) for x in (vertices,reference,initial,intrinsics,timestamps_s,
        outline_directions,outline_targets_px,outline_weights,depth_points_local,depth_target_m,depth_weights,support_planes)]
    v,ref,seed,k,t,axes,target,ow,dp,dz,dw,plane=arrays;lock=np.asarray(locked)
    n=len(ref);d=len(axes);j=dp.shape[1] if dp.ndim==3 else 0
    cfg=asdict(config)
    if (n<5 or ref.shape!=(n,4,4) or seed.shape!=ref.shape or k.shape!=(n,3,3) or t.shape!=(n,)
            or v.ndim!=2 or v.shape[1]!=3 or axes.shape!=(d,2) or d<8
            or target.shape!=(n,d) or ow.shape!=target.shape or dp.shape!=(n,j,3) or j<4
            or dz.shape!=(n,j) or dw.shape!=dz.shape or plane.shape!=(n,4)
            or lock.shape!=(n,) or lock.dtype!=bool or not lock.any()
            or not all(np.isfinite(x).all() for x in arrays)
            or np.any(ow<0) or np.any(dw<0) or np.any(ow.sum(1)<4) or np.any(dw.sum(1)<4)
            or np.any(np.diff(t)<=0) or not np.allclose(np.diff(t),np.diff(t)[0],rtol=1e-5,atol=1e-9)
            or not np.allclose(np.linalg.norm(plane[:,:3],axis=1),1,atol=1e-6)
            or not all(np.isfinite(x) and x>0 for x in cfg.values())
            or type(config.max_calls)!=int or type(config.proxy_vertices)!=int or config.proxy_vertices<8
            or progress_callback is not None and not callable(progress_callback)):
        raise ValueError('Complete finite native observations, uniform clock, explicit anchors and bounds required')
    for poses in (ref,seed):
        if (not np.allclose(poses[:,3],[0,0,0,1]) or not np.allclose(poses[:,:3,:3]@poses[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-5)
                or not np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-5)):
            raise ValueError('Rigid native reference and seed poses required')
    if not np.allclose(k[:,2],[0,0,1]) or np.any(k[:,[0,1],[0,1]]<=0):raise ValueError('Calibrated pinhole cameras required')
    hull=v[ConvexHull(v).vertices]
    ids=np.unique(np.r_[np.linspace(0,len(hull)-1,min(config.proxy_vertices,len(hull))).astype(int),hull.argmin(0),hull.argmax(0)])
    proxy=hull[ids];center=v.mean(0);centres=np.einsum('nij,j->ni',ref[:,:3,:3],center)+ref[:,:3,3]
    edit=np.flatnonzero(~lock);col=np.full(n,-1,int);col[edit]=np.arange(len(edit));width=d+j+7
    size=n*width+(n-2)*6;pattern=lil_matrix((size,len(edit)*6),dtype=np.int8)
    for i in edit:pattern[i*width:(i+1)*width,col[i]*6:(col[i]+1)*6]=1
    for i in range(n-2):
        for q in (i,i+1,i+2):
            if not lock[q]:pattern[n*width+i*6:n*width+(i+1)*6,col[q]*6:(col[q]+1)*6]=1
    bounds=np.tile(np.r_[np.repeat(config.translation_component_bound_m,3),np.repeat(np.deg2rad(config.rotation_component_bound_deg),3)],len(edit))
    seed_changes=np.c_[np.einsum('nij,j->ni',seed[:,:3,:3],center)+seed[:,:3,3]-centres,
        (Rotation.from_matrix(seed[:,:3,:3])*Rotation.from_matrix(ref[:,:3,:3]).inv()).as_rotvec()]
    x0=seed_changes[edit].ravel()
    if np.any(np.abs(x0)>=bounds):raise ValueError('Initial pose outside declared component uncertainty')
    dt=float(np.diff(t)[0]);started=time.monotonic();calls=0;best_x=x0.copy();best_cost=float('inf');budget=False;last_update=started
    def decode(x):
        changes=np.zeros((n,6));changes[edit]=x.reshape(-1,6)
        r=(Rotation.from_rotvec(changes[:,3:])*Rotation.from_matrix(ref[:,:3,:3])).as_matrix()
        trans=centres+changes[:,:3]-np.einsum('nij,j->ni',r,center)
        return r,trans,changes
    class Budget(Exception):pass
    def residual(x):
        nonlocal calls,best_x,best_cost,last_update
        if calls>=config.max_calls or time.monotonic()-started>config.max_seconds:raise Budget()
        calls+=1;r,trans,changes=decode(x)
        camera=np.einsum('mj,nkj->nmk',proxy,r)+trans[:,None,:]
        pixels=np.einsum('nmi,nji->nmj',camera,k);uv=pixels[:,:,:2]/np.maximum(pixels[:,:,2:],.01)
        extrema=np.max(uv@axes.T,axis=1)
        outline=(extrema-target)*np.sqrt(ow)/config.pixel_sigma
        depth=(np.einsum('nmi,ni->nm',dp,r[:,2,:])+trans[:,2,None]-dz)*np.sqrt(dw)/config.depth_sigma_m
        low=np.min(np.einsum('nmi,ni->nm',camera,plane[:,:3])+plane[:,3,None],axis=1)
        support=np.maximum(0,-low-config.support_penetration_m)/config.support_sigma_m
        prior=changes/np.r_[np.repeat(.30,3),np.repeat(np.deg2rad(30.),3)]
        frame=np.c_[outline,depth,support,prior].ravel()
        accel=np.diff(changes,n=2,axis=0)/dt**2
        accel[:,:3]/=config.acceleration_sigma_m_s2
        accel[:,3:]/=np.deg2rad(config.angular_acceleration_sigma_deg_s2)
        result=np.r_[frame,accel.ravel()]
        cost=float(np.sum(2*(np.sqrt(1+result*result)-1)))
        if cost<best_cost:best_cost=cost;best_x=x.copy()
        now=time.monotonic()
        if progress_callback is not None and now-last_update>=15:
            progress_callback(dict(function_calls=calls,seconds=now-started,best_proxy_cost=best_cost))
            last_update=now
        return result
    initial_cost=None
    try:
        residual(x0);initial_cost=best_cost
        least_squares(residual,x0,bounds=(-bounds,bounds),jac_sparsity=pattern.tocsr(),
            loss='soft_l1',f_scale=1.,max_nfev=config.max_calls,ftol=1e-5,xtol=1e-6,gtol=1e-5)
    except Budget:budget=True
    r,trans,changes=decode(best_x);poses=ref.copy();poses[:,:3,:3]=r;poses[:,:3,3]=trans;poses[lock]=ref[lock]
    return poses,dict(schema='real2sim2real.observed-pose-graph/1',config=cfg,function_calls=calls,
        budget_hit=budget,seconds=time.monotonic()-started,initial_proxy_cost=initial_cost,best_proxy_cost=best_cost,
        proxy_vertices=len(proxy),original_vertices=len(v),native_frames=n,locked_frames=np.flatnonzero(lock).tolist(),
        locked_poses_preserved=True,proposal_only=True,full_reference_approved=False,
        independent_native_triangle_depth_support_motion_checks_required=True,physics_validated=False)
