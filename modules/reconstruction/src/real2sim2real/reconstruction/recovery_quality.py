"""CPU quality and recovery proposals for RGB-derived object tracking.

No clean poses, learned policy or task-score input. A support plane is a nonpenetration
check within its footprint, never a constraint requiring an object to stay down.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Limits:
    track_iou: float = .55
    recovery_iou: float = .60
    precision: float = .60
    recall: float = .60
    depth_error_m: float = .10
    depth_points: int = 100
    support_penetration_m: float = .02
    step_translation_m: float = .08
    step_rotation_deg: float = 30.
    max_global_registrations: int = 12
    max_recovery_frames: int = 180


def configured_limits(recovery_frame_budget=None,*,source_frames=None,depth_error_m=None):
    """Finite work allowance and an optional explicitly recorded depth tolerance.

    Defaults preserve the original checks. A nondefault depth limit is a declared
    experiment, never an automatic fallback when a candidate fails.
    """
    limits=Limits()
    if depth_error_m is not None:
        if (isinstance(depth_error_m,bool) or not np.isfinite(depth_error_m)
                or not .01 <= depth_error_m <= .20):
            raise ValueError('Explicit depth-error limit must be within [0.01,0.20] metres')
        limits=replace(limits,depth_error_m=float(depth_error_m))
    if recovery_frame_budget is None:return limits
    if (type(recovery_frame_budget) is not int or recovery_frame_budget<1
        or (source_frames is not None and (type(source_frames) is not int or source_frames<2
            or recovery_frame_budget>source_frames-1))):
        raise ValueError('Recovery-frame budget must be a positive integer within the source update count')
    return replace(limits,max_recovery_frames=recovery_frame_budget)


def rigid(pose):
    pose = np.asarray(pose, float)
    return (pose.shape == (4, 4) and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(pose[:3, :3]@pose[:3, :3].T, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4))


def sample_vertices(vertices, spacing=.0015):
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite metric mesh vertices')
    _, indices = np.unique(np.round(vertices/spacing).astype(np.int64), axis=0, return_index=True)
    # Preserve coordinate extremes as well as voxel representatives.
    indices = np.unique(np.r_[indices, vertices.argmin(0), vertices.argmax(0)])
    return vertices[indices]


def depth_residual_summary(signed_residuals_m):
    """Describe residual spread without changing depth acceptance or direction.

    A small signed median is not evidence of small absolute error. Report the
    distribution so mixed signs cannot be mistaken for a reliable correction.
    Residuals are observed minus rendered camera depth, in metres.
    """
    x=np.asarray(signed_residuals_m,float)
    if x.ndim!=1 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional metre residuals required')
    return dict(depth_points=int(len(x)),
        depth_error_m=float(np.median(np.abs(x))) if len(x) else None,
        depth_shift_m=float(np.median(x)) if len(x) else None,
        depth_residual_quantiles_m=np.quantile(x,[.1,.25,.5,.75,.9]).tolist() if len(x) else None,
        depth_positive_fraction=float(np.mean(x>0)) if len(x) else None,
        depth_negative_fraction=float(np.mean(x<0)) if len(x) else None)


def consistency(vertices, pose, K, depth, observed, support):
    if not rigid(pose):
        raise ValueError('Nonrigid candidate')
    K = np.asarray(K, float)
    depth = np.asarray(depth)
    observed = np.asarray(observed, bool)
    if K.shape != (3, 3) or not np.isfinite(K).all() or observed.shape != depth.shape:
        raise ValueError('Incompatible camera/depth/mask')
    camera = np.asarray(vertices)@pose[:3, :3].T+pose[:3, 3]
    positive = camera[:, 2] > .01
    uv = camera[positive]@K.T
    uv = uv[:, :2]/uv[:, 2, None]
    pixels = np.rint(uv).astype(int)
    h, w = depth.shape
    valid = (pixels[:, 0]>=0)&(pixels[:, 0]<w)&(pixels[:, 1]>=0)&(pixels[:, 1]<h)
    flat = pixels[valid, 1]*w+pixels[valid, 0]
    # Sparse z-buffer: allocate only occupied pixels, not a full image per pose.
    unique, inverse = np.unique(flat, return_inverse=True)
    z = np.full(len(unique), np.inf)
    np.minimum.at(z, inverse, camera[positive, 2][valid])
    measured = depth.ravel()[unique]
    usable = observed.ravel()[unique] & np.isfinite(measured) & (measured>.001) & (measured<8.)
    signed = measured[usable]-z[usable]
    y, x = np.where(observed)
    if len(unique) and len(x):
        displacement = [float(x.mean()-(unique%w).mean()), float(y.mean()-(unique//w).mean())]
    else:
        displacement = None
    plane = np.asarray(support['plane_camera'], float)
    center = np.asarray(support['center_camera_m'], float)
    if (plane.shape != (4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-5)
            or center.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(support['radius_m']) or support['radius_m'] <= 0):
        raise ValueError('Invalid support candidate')
    delta = camera-center
    radial = delta-np.outer(delta@plane[:3], plane[:3])
    over_support = np.linalg.norm(radial, axis=1) <= support['radius_m']
    heights = camera@plane[:3]+plane[3]
    return dict(**depth_residual_summary(signed),
        centroid_shift_px=displacement,
        support_penetration_m=max(0., -float(heights[over_support].min())) if over_support.any() else 0.,
        behind_camera=bool(not positive.all()))


def rejection_reasons(pose, previous, metrics, geometry, *, recovery=False, limits=Limits()):
    if not rigid(pose) or not rigid(previous):
        return ['nonrigid']
    reasons = []
    for key, minimum in (('iou', limits.recovery_iou if recovery else limits.track_iou),
                         ('precision', limits.precision), ('recall', limits.recall)):
        value = metrics.get(key)
        if value is None or not np.isfinite(value) or not minimum <= value <= 1.:
            reasons.append(key)
    if geometry.get('behind_camera', True):
        reasons.append('camera_plane')
    value = geometry.get('depth_error_m')
    if (geometry.get('depth_points', 0)<limits.depth_points or value is None
            or not np.isfinite(value) or not 0 <= value <= limits.depth_error_m):
        reasons.append('depth')
    penetration = geometry.get('support_penetration_m')
    if penetration is None or not np.isfinite(penetration) or not 0<=penetration<=limits.support_penetration_m:
        reasons.append('support')
    if np.linalg.norm(pose[:3, 3]-previous[:3, 3]) > limits.step_translation_m:
        reasons.append('translation_step')
    angle = np.rad2deg(Rotation.from_matrix(pose[:3, :3]@previous[:3, :3].T).magnitude())
    if angle > limits.step_rotation_deg:
        reasons.append('rotation_step')
    return reasons


def local_proposals(previous, current, K, geometry):
    """Bounded pose hypotheses, not accepted predictions or support projection."""
    K = np.asarray(K)
    if not rigid(previous) or not rigid(current):
        raise ValueError('Local recovery requires rigid prior and tracker poses')
    delta_depth = geometry.get('depth_shift_m')
    delta_pixel = geometry.get('centroid_shift_px')
    if delta_depth is None or delta_pixel is None or not np.isfinite([delta_depth, *delta_pixel]).all():
        return np.stack([previous, current])
    # Proposals cross a float32 tensor boundary before the unchanged quality
    # check. Leave room for coordinate-dependent quantization rather than
    # generating an exactly-on-limit displacement that can round outside it.
    coordinate_scale = max(1., float(np.max(np.abs(previous[:3, 3]))),
                           float(np.max(np.abs(current[:3, 3]))))
    bound = max(0., Limits().step_translation_m -
                8*np.finfo(np.float32).eps*coordinate_scale)
    correction = current[:3, 3].astype(float)/float(current[2, 3]) * np.clip(delta_depth, -.08, .08)
    correction[:2] += np.clip(delta_pixel, -16, 16)*current[2, 3]/K.diagonal()[:2]
    norm = np.linalg.norm(correction)
    if norm > bound:
        correction *= bound/norm
    proposals = [previous.copy(), current.copy()]
    for base in (previous, current):
        for fraction in (.5, 1.):
            pose = base.copy(); pose[:3, 3] += fraction*correction
            proposals.append(pose)
    # Explore local angular alternatives around the last accepted orientation.
    for axis in range(3):
        for angle in (-15., 15.):
            rotation = np.zeros(3); rotation[axis] = np.deg2rad(angle)
            pose = previous.copy()
            pose[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()@previous[:3, :3]
            pose[:3, 3] += correction
            proposals.append(pose)
    return np.stack(proposals)


def pixel_translation_proposals(pose,K,*,radius_px=1.):
    """At most25 unaccepted subpixel translations; preserve rotation and depth.

    This searches around a camera pose, not around a temporal reference. The
    consumer MUST recheck every result against the actual previous accepted pose
    and the independent image/depth/support gates.
    """
    p=np.asarray(pose,float);K=np.asarray(K,float)
    if (not rigid(p) or p[2,3]<=.01 or K.shape!=(3,3) or not np.isfinite(K).all()
        or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]) or abs(np.linalg.det(K))<1e-12
        or isinstance(radius_px,bool) or not np.isfinite(radius_px) or not 0<radius_px<=2):
        raise ValueError('Rigid positive-depth pose, calibrated camera and explicit pixel radius in(0,2] required')
    out=[]
    for y in np.linspace(-radius_px,radius_px,5):
        for x in np.linspace(-radius_px,radius_px,5):
            q=p.copy();q[:3,3]+=np.linalg.solve(K,np.array([x*p[2,3],y*p[2,3],0.]));out.append(q)
    return np.asarray(out)


def protocol(*,recovery_frame_budget=None,depth_error_m=None):
    return dict(schema='real2sim2real.tracking-recovery/1', limits=asdict(configured_limits(recovery_frame_budget,depth_error_m=depth_error_m)),
        observed_depth='own monocular estimate, not ground truth',
        support='reject penetration only inside the reviewed support footprint; lifting is allowed',
        recovery='local hypotheses, then at most one depth and one RGB registration per failing frame',
        failure='stop; no unvalidated carry-forward or threshold relaxation')
