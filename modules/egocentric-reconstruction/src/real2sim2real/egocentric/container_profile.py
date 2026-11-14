"""Bounded shared exterior-profile fitting; hidden wall geometry remains unknown."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .camera import transform, rigid, intrinsic

KNOTS = np.array([0., .25, .75, 1.])


def validate(profile):
    a = np.asarray(profile, float)
    if a.shape != (5,) or not np.isfinite(a).all() or not ((a[:4]>0)&(a[:4]<.2)).all() or not 0<a[4]<1:
        raise ValueError("Four positive metric radii and positive height required")
    return a


def distance(points, profile):
    a=validate(profile);p=np.asarray(points,float)
    r=np.linalg.norm(p[:,:2],axis=1);z=p[:,2];levels=KNOTS*a[4]
    distances=[np.hypot(np.maximum(r-a[0],0),z)]
    for i in range(3):
        dr=a[i+1]-a[i];dz=levels[i+1]-levels[i]
        t=np.clip(((r-a[i])*dr+(z-levels[i])*dz)/(dr*dr+dz*dz),0,1)
        distances.append(np.hypot(r-a[i]-t*dr,z-levels[i]-t*dz))
    return np.min(distances,axis=0)


def surface(profile, rings=24, segments=96):
    a=validate(profile);z=np.linspace(0,a[4],rings);r=np.interp(z,KNOTS*a[4],a[:4]);angle=np.linspace(0,2*np.pi,segments,endpoint=False)
    return np.stack(np.broadcast_arrays(r[:,None]*np.cos(angle),r[:,None]*np.sin(angle),z[:,None]),-1).reshape(-1,3)


def linear_profile(dimensions):
    rb,rt,h=dimensions
    return validate(np.r_[rb+(rt-rb)*KNOTS,h])


def residual_breakdown(points_local, profile):
    """Untrimmed fit diagnostics by axial region; never an admission filter.

    Region labels depend on the candidate pose/profile. Signed radial residuals
    are reported only on the middle wall, where a bottom-cap explanation is less
    likely. These diagnostics cannot isolate stereo, registration or shape error.
    """
    a=validate(profile);p=np.asarray(points_local,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError("Finite object-space points required")
    fraction=p[:,2]/a[4];errors=distance(p,a)
    regions={'outside_height':(fraction<0)|(fraction>1),
             'bottom_band':(fraction>=0)&(fraction<.15),
             'middle_wall':(fraction>=.15)&(fraction<=.85),
             'rim_band':(fraction>.85)&(fraction<=1)}
    report={}
    for name,mask in regions.items():
        values=errors[mask]
        row=dict(points=int(mask.sum()),median_m=float(np.median(values)) if len(values) else None,
                 p90_m=float(np.quantile(values,.9)) if len(values) else None,
                 over_10mm_points=int((values>.01).sum()))
        if name=='middle_wall' and len(values):
            radial=np.linalg.norm(p[mask,:2],axis=1)-np.interp(p[mask,2],KNOTS*a[4],a[:4])
            row['signed_radial_q10_q50_q90_m']=np.quantile(radial,[.1,.5,.9]).tolist()
        report[name]=row
    return report


def fit_shared(views, initial_profile, *, radius_bounds=(.025,.075), height_bounds=(.12,.23)):
    """Only supplied training views enter optimization. No validation inputs.

    Each view supplies fit-only points, intrinsics, bottom/rim pixels and their
    uncertainties, and an initial camera_from_object pose. Endpoint costs and
    depth costs retain the endpoint adapter's normalization. Weak profile priors
    discourage unsupported bulges; they are assumptions, not measurements.
    """
    initial=validate(initial_profile)
    if len(views)<2:raise ValueError("At least two training views required")
    starts=[]; packing=[]
    for v in views:
        p=np.asarray(v['points'],float);uv=np.asarray(v['pixels'],float);sigma=np.asarray(v['sigma'],float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<30 or not np.isfinite(p).all() or uv.shape!=(2,2) or not np.isfinite(uv).all() or sigma.shape!=(2,) or not np.isfinite(sigma).all() or (sigma<=0).any():raise ValueError("Finite points and uncertain endpoint observations required")
        intrinsic(v['k']);t=rigid(v['pose'])
        mode=v.get('pose_mode','free')
        if mode not in ['free','fixed','support_translation']:
            raise ValueError("Known profile-training pose mode required")
        visible=np.asarray(v.get('endpoint_visible',[True,True]))
        if visible.shape!=(2,) or visible.dtype.kind!='b':
            raise ValueError("Two explicit endpoint visibility booleans required")
        start=np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_rotvec()] if mode=='free' else np.zeros(2) if mode=='support_translation' else np.empty(0)
        packing.append((5+len(starts),len(start),mode));starts.extend(start)
    def pose(x):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix();t[:3,3]=x[:3];return t
    def view_pose(x,j):
        offset,count,mode=packing[j]
        if mode=='free':return pose(x[offset:offset+count])
        t=np.array(views[j]['pose'],float,copy=True)
        if mode=='support_translation':
            t[:3,3]+=t[:3,:2]@x[offset:offset+count]
        return t
    def residual(x):
        a=x[:5];parts=[]
        for j,v in enumerate(views):
            t=view_pose(x,j);p=np.asarray(v['points']);local=transform(p,np.linalg.inv(t));parts.append(distance(local,a)/.003/np.sqrt(len(p)))
            ends=transform(np.array([[0.,0,0],[0,0,a[4]]]),t);q=ends@np.asarray(v['k']).T
            endpoint_error=(q[:,:2]/np.maximum(q[:,2:],.01)-v['pixels'])/np.asarray(v['sigma'])[:,None]
            parts.append((endpoint_error*np.asarray(v.get('endpoint_visible',[True,True]))[:,None]).ravel())
            parts.append(np.maximum(.01-ends[:,2],0)*100)
            if v.get('image_constraints'):
                from .container_contours import contour_residuals
                parts.append(contour_residuals(a,t,v['k'],v['image_constraints']))
        line=a[0]+(a[3]-a[0])*KNOTS
        parts.extend([(a[:4]-line)/.01,(a-initial)/np.array([.03]*4+[.08])])
        return np.concatenate(parts)
    lo=list([radius_bounds[0]]*4+[height_bounds[0]]);hi=list([radius_bounds[1]]*4+[height_bounds[1]])
    for v,(_,_,mode) in zip(views,packing):
        if mode=='free':
            c=np.asarray(v['pose'])[:3,3];lo.extend(np.r_[c-.4,[-6.4]*3]);hi.extend(np.r_[c+.4,[6.4]*3])
        elif mode=='support_translation':
            lo.extend([-.03,-.03]);hi.extend([.03,.03])
    solution=least_squares(residual,np.r_[initial,starts],bounds=(lo,hi),loss='soft_l1',f_scale=1,max_nfev=240)
    return dict(profile_m=solution.x[:5].tolist(),optimizer_success=bool(solution.success),
                bound_hit=bool(np.any(np.isclose(solution.x[:5],lo[:5],atol=1e-5)|np.isclose(solution.x[:5],hi[:5],atol=1e-5))),
                training_poses=[view_pose(solution.x,j).tolist() for j in range(len(views))],
                training_pose_modes=[v.get('pose_mode','free') for v in views],
                training_views=len(views),cost=float(solution.cost),wall_thickness_verified=False,control_ready=False)
