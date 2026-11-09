"""Bounded fixed-mesh pose proposals from a visible native image outline.

The convex support function ignores holes and concavities. This is deliberately
an initializer, never a tracker or an acceptance gate. Validate the original
triangles, source appearance, depth predictions and temporal seams separately.
"""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def propose(vertices,rotation,translation,intrinsics,mask,*,human_mask=None,
            max_rotation_deg=20.,max_center_shift_m=.04,max_calls=100,directions=32,
            preserve_camera_center_depth=True):
    v,r,t,k=map(lambda x:np.asarray(x,float),(vertices,rotation,translation,intrinsics))
    mask=np.asarray(mask);human=np.zeros_like(mask) if human_mask is None else np.asarray(human_mask)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or r.shape!=(3,3) or t.shape!=(3,) or k.shape!=(3,3)
        or mask.ndim!=2 or mask.dtype!=bool or human.shape!=mask.shape or human.dtype!=bool
        or not all(np.isfinite(x).all() for x in (v,r,t,k))
        or not np.allclose(r@r.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1.,atol=1e-5)
        or not np.allclose(k[2],[0,0,1]) or min(k[0,0],k[1,1])<=0
        or any(isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in (max_rotation_deg,max_center_shift_m))
        or max_rotation_deg>45 or type(max_calls)!=int or max_calls<1 or type(directions)!=int or directions<8
        or type(preserve_camera_center_depth)!=bool):
        raise ValueError('Finite metric mesh/rigid pose/camera, boolean native masks and explicit small bounds required')
    yy,xx=np.nonzero(mask)
    if len(xx)<100:raise ValueError('Insufficient visible outline evidence')
    overlap=np.count_nonzero(mask&human)/len(xx)
    if overlap>.15:raise ValueError('Human overlap makes convex-outline fitting unsafe')
    boundary=mask&~binary_erosion(mask)
    y,x=np.nonzero(boundary)
    observed=np.c_[x,y].astype(float)
    angles=np.linspace(0,2*np.pi,directions,endpoint=False);axes=np.c_[np.cos(angles),np.sin(angles)]
    target=np.max(observed@axes.T,axis=0)
    # A full mesh convex hull preserves all projected convex extrema. It is a
    # fitting proxy only, never exported in place of the source triangle mesh.
    hull=v[ConvexHull(v).vertices];center=v.mean(0);camera_center=r@center+t
    angle=np.deg2rad(max_rotation_deg)
    dimension=2 if preserve_camera_center_depth else 3
    bounds=np.r_[np.repeat(angle/np.sqrt(3),3),np.repeat(max_center_shift_m/np.sqrt(dimension),dimension)]
    def displacement(parameters):return np.r_[parameters[3:],0.] if preserve_camera_center_depth else parameters[3:]
    def pose(parameters):
        rr=Rotation.from_rotvec(parameters[:3]).as_matrix()@r
        return rr,camera_center+displacement(parameters)-rr@center
    def residual(parameters):
        rr,tt=pose(parameters);points=hull@rr.T+tt
        if np.any(points[:,2]<=.01):return np.full(directions+len(bounds),1e6)
        uv=points@k.T;uv=uv[:,:2]/uv[:,2:]
        error=np.max(uv@axes.T,axis=0)-target
        return np.r_[error,parameters[:3]/angle*.25,parameters[3:]/max_center_shift_m*.25]
    result=least_squares(residual,np.zeros(len(bounds)),bounds=(-bounds,bounds),max_nfev=max_calls,
        diff_step=1e-4,loss='soft_l1',f_scale=2.,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    rr,tt=pose(result.x)
    return rr,tt,dict(schema='real2sim2real.outline-pose-proposal/1',proposal_only=True,candidate_approved=False,
        objective='Convex directional outline support, with small pose-change regularization',
        full_mesh_vertices=len(v),convex_proxy_vertices=len(hull),human_overlap_fraction=float(overlap),
        support_rmse_before_px=float(np.sqrt(np.mean(residual(np.zeros(len(bounds)))[:directions]**2))),
        support_rmse_after_px=float(np.sqrt(np.mean(residual(result.x)[:directions]**2))),
        rotation_change_deg=float(np.rad2deg(np.linalg.norm(result.x[:3]))),
        center_shift_m=float(np.linalg.norm(result.x[3:])),camera_center_depth_preserved=preserve_camera_center_depth,
        camera_center_depth_residual_m=float(abs((rr@center+tt)[2]-camera_center[2])),
        mesh_changed=False,metric_scale_changed=False,depth_used_for_fitting=False,
        native_triangle_appearance_depth_and_temporal_validation_required=True,
        optimizer_calls=int(result.nfev),max_rotation_deg=max_rotation_deg,max_center_shift_m=max_center_shift_m)


def select_observed_path(poses,costs,usable,previous,*,max_step_m=.08,max_rotation_deg=30.,
                         reference_poses=None,max_correction_step_deg=None,correction_weight=0.):
    """Select measured per-frame proposals; never interpolate missing evidence."""
    p,c,ok,prior=np.asarray(poses,float),np.asarray(costs,float),np.asarray(usable),np.asarray(previous,float)
    if (p.ndim!=4 or p.shape[2:]!=(4,4) or not len(p) or c.shape!=p.shape[:2] or ok.shape!=c.shape
        or ok.dtype!=bool or prior.shape!=(4,4) or not np.isfinite(p).all() or not np.isfinite(c).all()
        or not np.isfinite(prior).all() or min(max_step_m,max_rotation_deg)<=0):raise ValueError('Complete finite observed pose/cost/boolean evidence grid required')
    all_poses=np.concatenate([p.reshape(-1,4,4),prior[None]])
    if (not np.allclose(all_poses[:,3],[0,0,0,1]) or not np.allclose(all_poses[:,:3,:3]@all_poses[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-5)
        or not np.allclose(np.linalg.det(all_poses[:,:3,:3]),1,atol=1e-5)):raise ValueError('Rigid observed poses required')
    n,m=c.shape;back=np.zeros((n,m),int);scores=np.array([0.]);last=prior[None]
    reference=None
    if reference_poses is not None:
        reference=np.asarray(reference_poses,float)
        if (reference.shape!=(n,4,4) or not np.isfinite(reference).all()
            or not np.allclose(reference[:,3],[0,0,0,1])
            or not np.allclose(reference[:,:3,:3]@reference[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-5)
            or not np.allclose(np.linalg.det(reference[:,:3,:3]),1,atol=1e-5)
            or max_correction_step_deg is None or not np.isfinite(max_correction_step_deg) or max_correction_step_deg<=0
            or not np.isfinite(correction_weight) or correction_weight<0):raise ValueError('Explicit reference and positive correction-step bound required')
    elif max_correction_step_deg is not None or correction_weight!=0:raise ValueError('Correction limits require a reference sequence')
    last_correction=np.eye(3)[None]
    for i in range(n):
        displacement=np.linalg.norm(last[:,None,:3,3]-p[i,None,:,:3,3],axis=-1)
        relative=last[:,None,:3,:3]@p[i,None,:,:3,:3].transpose(0,1,3,2)
        angular=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
        transitions=np.where((displacement<=max_step_m)&(angular<=max_rotation_deg),
            .001*(angular/max_rotation_deg)**2+.001*(displacement/max_step_m)**2,np.inf)
        if reference is not None:
            correction=p[i,:,:3,:3]@reference[i,:3,:3].T
            delta=last_correction[:,None]@correction[None].transpose(0,1,3,2)
            degrees=np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
            transitions+=np.where(degrees<=max_correction_step_deg,
                correction_weight*(degrees/max_correction_step_deg)**2,np.inf)
        candidate=scores[:,None]+transitions
        back[i]=candidate.argmin(0);scores=np.where(ok[i],candidate.min(0)+c[i],np.inf)
        if not np.isfinite(scores).any():raise ValueError('No admissible observed path at local frame '+str(i))
        last=p[i]
        if reference is not None:last_correction=correction
    states=np.zeros(n,int);states[-1]=int(scores.argmin())
    for i in range(n-1,0,-1):states[i-1]=back[i,states[i]]
    return p[np.arange(n),states],dict(selected_indices=states.tolist(),interpolation_used=False,
        candidate_approved=False,native_gpu_validation_required=True,max_step_m=max_step_m,max_rotation_deg=max_rotation_deg,
        max_correction_step_deg=max_correction_step_deg,correction_weight=correction_weight)


def review_motion(reference,proposal,*,max_p95_ratio=1.25,p95_slack_deg=1.,high_step_deg=10.,max_extra_high_steps=5):
    """Compare native-step rotational roughness; include the preceding pose.

    These caller-visible limits are a proposal screen, not metric truth or a
    claim that the parent motion is correct. Per-frame and support checks still
    apply. A rigid per-frame renderer pass alone cannot establish smooth motion.
    """
    old,new=np.asarray(reference,float),np.asarray(proposal,float)
    limits=np.asarray([max_p95_ratio,p95_slack_deg,high_step_deg],float)
    if (old.ndim!=3 or old.shape[1:]!=(4,4) or len(old)<2 or new.shape!=old.shape
        or not np.isfinite(old).all() or not np.isfinite(new).all() or not np.isfinite(limits).all()
        or max_p95_ratio<1 or p95_slack_deg<0 or high_step_deg<=0 or type(max_extra_high_steps)!=int or max_extra_high_steps<0):
        raise ValueError('Matching finite pose sequences and explicit native-step review limits required')
    results=[]
    for sequence in [old,new]:
        if not np.allclose(sequence[:,:3,:3]@sequence[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-5):raise ValueError('Rigid rotations required')
        angle=np.rad2deg((Rotation.from_matrix(sequence[1:,:3,:3])*Rotation.from_matrix(sequence[:-1,:3,:3]).inv()).magnitude())
        results.append(dict(p95_rotation_step_deg=float(np.quantile(angle,.95)),high_rotation_step_count=int((angle>high_step_deg).sum()),max_rotation_step_deg=float(angle.max())))
    a,b=results
    passed=b['p95_rotation_step_deg']<=a['p95_rotation_step_deg']*max_p95_ratio+p95_slack_deg and b['high_rotation_step_count']<=a['high_rotation_step_count']+max_extra_high_steps
    return dict(passed=passed,before=a,after=b,limits=dict(max_p95_ratio=max_p95_ratio,p95_slack_deg=p95_slack_deg,high_step_deg=high_step_deg,max_extra_high_steps=max_extra_high_steps),
        per_frame_renderer_pass_is_motion_approval=False,reference_is_ground_truth=False,full_reference_approved=False)
