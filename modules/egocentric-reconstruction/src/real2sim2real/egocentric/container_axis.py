"""Endpoint-constrained pose and yaw-invariant protrusion exclusion hypotheses."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .camera import intrinsic, rigid, transform
from .tapered_container import outer_distance


def endpoints(pose, height, k):
    p = transform(np.array([[0., 0, 0], [0, 0, height]]), rigid(pose))
    if (p[:, 2] <= 0).any():
        raise ValueError("Endpoint behind camera")
    q = p @ intrinsic(k).T
    return q[:, :2] / q[:, 2:]


def fit_endpoint_pose(points, dimensions, k, pixels, uncertainty_px, initial_pose, *, surface_distance=None,
                      image_constraints=None, projection_profile=None):
    """Dimensions fixed; endpoints ordered bottom then rim, with explicit uncertainty.

    Endpoint reprojection errors are fitting residuals, not independent labels.
    The caller must retain separate depth/visibility checks and occlusion flags.
    """
    p, uv, sigma = np.asarray(points, float), np.asarray(pixels, float), np.asarray(uncertainty_px, float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p)<30 or not np.isfinite(p).all()
            or uv.shape != (2,2) or not np.isfinite(uv).all()
            or sigma.shape != (2,) or not np.isfinite(sigma).all() or (sigma<=0).any()):
        raise ValueError("Finite observations and positive endpoint uncertainties required")
    initial = rigid(initial_pose)
    distance = outer_distance if surface_distance is None else surface_distance
    centroid = np.median(p, axis=0)
    # Both endpoint interpretations are optimized, rather than carrying forward
    # a potentially inverted depth-only solution as the sole seed.
    flipped = initial.copy()
    flipped[:3, 3] += initial[:3, 2] * dimensions[2]
    flipped[:3, :3] = initial[:3, :3] @ Rotation.from_euler('x',180,degrees=True).as_matrix()
    def matrix(x):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix();t[:3,3]=x[:3];return t
    def residual(x):
        t=matrix(x)
        depth=distance(transform(p,np.linalg.inv(t)),dimensions)/.003/np.sqrt(len(p))
        camera_points=transform(np.array([[0.,0,0],[0,0,dimensions[2]]]),t)
        q=camera_points@intrinsic(k).T
        image=q[:,:2]/np.maximum(q[:,2:],.01)
        parts=[depth,((image-uv)/sigma[:,None]).ravel(),np.maximum(.01-camera_points[:,2],0)*100]
        if image_constraints:
            from .container_contours import contour_residuals
            from .container_profile import linear_profile
            profile=linear_profile(dimensions) if projection_profile is None else projection_profile
            parts.append(contour_residuals(profile,t,k,image_constraints))
        return np.concatenate(parts)
    candidates=[]
    for t in [initial,flipped]:
        start=np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_rotvec()]
        lo=np.r_[np.minimum(centroid-.5,start[:3]-.001),[-6.4]*3];hi=np.r_[np.maximum(centroid+.5,start[:3]+.001),[6.4]*3]
        s=least_squares(residual,start,bounds=(lo,hi),loss='soft_l1',f_scale=1,max_nfev=160)
        candidates.append(s)
    best=min(candidates,key=lambda s:s.cost)
    pose=matrix(best.x); projected=endpoints(pose,dimensions[2],k)
    return pose,dict(optimizer_success=bool(best.success),endpoint_fit_error_px=np.linalg.norm(projected-uv,axis=1).tolist(),
                     projected_endpoints_px=projected.tolist(),yaw_observable=False,
                     visible_contours_used_for_pose=bool(image_constraints),
                     endpoint_fit_is_independent_validation=False)


def axis_transition(first, second, delta_s):
    """Compare directed axes in one fixed world; yaw intentionally excluded."""
    a,b=rigid(first),rigid(second)
    if not np.isfinite(delta_s) or delta_s<=0:
        raise ValueError("Positive measured time interval required")
    angle=float(np.rad2deg(np.arccos(np.clip(a[:3,2]@b[:3,2],-1,1))))
    return dict(axis_angle_deg=angle,mean_axis_rate_deg_s=angle/delta_s,
                bottom_translation_m=float(np.linalg.norm(b[:3,3]-a[:3,3])),
                interpretation='sparse endpoint displacement; not maximum instantaneous velocity or a continuity certificate')


def exclusion_bands(points_local, dimensions, *, axial_margin_m=.015, radial_margin_m=.015):
    """Yaw-invariant no-grasp bands from uncertain protrusion samples.

    Do not infer empty regions from absent samples. Rotation about an
    axisymmetric body cannot reliably localize an azimuthal protrusion.
    """
    p=np.asarray(points_local,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or min(axial_margin_m,radial_margin_m)<=0:
        raise ValueError("Finite samples and positive conservative margins required")
    if not len(p):
        return dict(available=False,unobserved_protrusions_excluded=False)
    return dict(available=True,z_min_m=float(p[:,2].min()-axial_margin_m),
                z_max_m=float(p[:,2].max()+axial_margin_m),
                outer_radius_m=float(max(np.linalg.norm(p[:,:2],axis=1).max(),max(dimensions[:2]))+radial_margin_m),
                azimuth='all',geometry_verified=False,unobserved_protrusions_excluded=False)
