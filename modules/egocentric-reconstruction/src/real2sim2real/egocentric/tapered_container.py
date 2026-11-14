"""Explicit axisymmetric outer-envelope hypothesis, not recovered wall thickness."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def surface(radius_bottom, radius_top, height, rings=16, segments=96):
    if not (0 < radius_bottom < 0.2 and 0 < radius_top < 0.2 and 0 < height < 1):
        raise ValueError("Positive bounded metric dimensions required")
    z = np.linspace(0, height, rings)
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    radius = radius_bottom + (radius_top - radius_bottom) * z / height
    return np.stack(np.broadcast_arrays(radius[:, None]*np.cos(angle),
                                       radius[:, None]*np.sin(angle), z[:, None]), -1).reshape(-1, 3)


def outer_distance(points, dimensions):
    """Unsigned distance approximation to tapered side and bottom; mouth stays open."""
    rb, rt, h = dimensions
    surface(rb, rt, h, rings=2, segments=3)  # validate dimensions
    p = np.asarray(points, float)
    radial = np.linalg.norm(p[:, :2], axis=1)
    slope = (rt-rb)/h
    z = np.clip((p[:, 2] + slope*(radial-rb))/(1+slope*slope), 0, h)
    side = np.hypot(radial-(rb+slope*z), p[:, 2]-z)
    bottom = np.hypot(np.maximum(radial-rb, 0), p[:, 2])
    return np.minimum(side, bottom)


def projected_outline(points, camera_from_world, k):
    p = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    if (p[:, 2] <= 0).any():
        raise ValueError("Hypothesis behind camera")
    uv = p @ k.T
    return cv2.convexHull((uv[:, :2]/uv[:, 2:]).astype(np.float32)).reshape(-1, 2)


def outline_samples(polygon, count=180):
    p = np.asarray(polygon, float)
    q = np.roll(p, -1, axis=0)
    lengths = np.linalg.norm(q-p, axis=1)
    if len(p)<3 or not np.isfinite(p).all() or lengths.sum() <= 0:
        raise ValueError("Finite nondegenerate outline required")
    distance = np.linspace(0, lengths.sum(), count, endpoint=False)
    cumulative = np.r_[0, np.cumsum(lengths)]
    idx = np.minimum(np.searchsorted(cumulative, distance, side='right')-1, len(p)-1)
    t = (distance-cumulative[idx])/np.maximum(lengths[idx], 1e-12)
    return p[idx]+t[:, None]*(q[idx]-p[idx])


def fit_resting(points, outline, camera_from_world, k):
    """Assume axis normal to level table and bottom at z=0; fit one visible view."""
    points = np.asarray(points, float)
    if len(points)<30 or not np.isfinite(points).all():
        raise ValueError("At least 30 finite outer-wall observations required")
    observed = outline_samples(outline)
    center = np.median(points[:, :2], axis=0)
    def residual(x):
        shape = surface(*x[2:]) + [x[0],x[1],0]
        projected = outline_samples(projected_outline(shape,camera_from_world,k))
        silhouette = np.r_[cKDTree(observed).query(projected)[0],cKDTree(projected).query(observed)[0]]
        local = points-[x[0],x[1],0]
        return np.r_[outer_distance(local,x[2:]),silhouette*0.001]
    lo=np.r_[center-0.09,0.025,0.025,0.10];hi=np.r_[center+0.09,0.095,0.095,0.30]
    result=least_squares(residual,np.r_[center,0.045,0.06,0.20],bounds=(lo,hi),loss='soft_l1',f_scale=0.003,max_nfev=180)
    return dict(center_xy_m=result.x[:2].tolist(), dimensions_m=result.x[2:].tolist(),
                optimizer_success=bool(result.success),bound_hit=bool(np.any(np.isclose(result.x,lo,atol=1e-5)|np.isclose(result.x,hi,atol=1e-5))),
                assumptions=['axisymmetric tapered exterior','bottom on table','axis upright'],
                wall_thickness_verified=False,control_ready=False)


def fit_pose(points, dimensions):
    """Fit rigid pose only; dimensions frozen. No reference trajectory used."""
    p=np.asarray(points,float)
    if len(p)<30 or not np.isfinite(p).all():
        raise ValueError("Insufficient finite pose observations")
    centroid=np.median(p,axis=0)
    _,_,axes=np.linalg.svd(p-centroid,full_matrices=False)
    starts=[]
    for axis in axes:
        for sign in [-1,1]:
            z=sign*axis; x=np.cross(z,np.eye(3)[np.argmin(abs(z))]);x/=np.linalg.norm(x)
            rotation=np.column_stack([x,np.cross(z,x),z])
            starts.append(np.r_[centroid-z*dimensions[2]/2,Rotation.from_matrix(rotation).as_rotvec()])
    def residual(x):
        local=(p-x[:3])@Rotation.from_rotvec(x[3:]).as_matrix()
        return outer_distance(local,dimensions)
    solutions=[least_squares(residual,s,loss='soft_l1',f_scale=0.003,max_nfev=100,
                            bounds=(np.r_[centroid-0.4,[-6.4]*3],np.r_[centroid+0.4,[6.4]*3])) for s in starts]
    best=min(solutions,key=lambda s:float(np.median(residual(s.x))))
    pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(best.x[3:]).as_matrix();pose[:3,3]=best.x[:3]
    return pose,dict(optimizer_success=bool(best.success),fit_median_m=float(np.median(residual(best.x))),
                     orientation_about_symmetry_axis_observable=False)
