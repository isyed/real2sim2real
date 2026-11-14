"""Bounded cross-view registration hypotheses and flow-only camera compensation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .camera import rigid, intrinsic
from .temporal_alignment import project


def _observations(points,pixels):
    p,q=np.asarray(points,float),np.asarray(pixels,float)
    if p.ndim!=2 or p.shape[1]!=3 or q.shape!=(len(p),2) or len(p)<12 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('At least twelve finite paired 3D/2D static observations required')
    return p,q


def _delta(x):
    t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();t[:3,3]=x[3:];return t


def fit_shared_registration(views,initial,k,*,rotation_bound_deg=2,translation_bound_m=.02):
    """Fit one six-parameter correction from explicitly supplied training views.

    Each view contains points/pixels only from its declared fit subset. Test views
    are never supplied. Pixel residuals have equal weight; robust loss is 1.5 px.
    No calibration file is modified, and timing error can still bias this estimate.
    """
    initial=rigid(initial);k=intrinsic(k)
    if len(views)<2 or not 0<rotation_bound_deg<=5 or not 0<translation_bound_m<=.05:
        raise ValueError('Multiple training views and small explicit trust bounds required')
    observed=[_observations(v['points'],v['pixels']) for v in views]
    bounds=np.r_[[np.deg2rad(rotation_bound_deg)]*3,[translation_bound_m]*3]
    def residual(x):
        t=_delta(x)@initial
        return np.concatenate([(project(p,t,k)-q).ravel() for p,q in observed])
    result=least_squares(residual,np.zeros(6),bounds=(-bounds,bounds),loss='soft_l1',f_scale=1.5,max_nfev=200)
    return _delta(result.x)@initial,dict(optimizer_success=bool(result.success),bound_hit=bool(np.any(np.isclose(abs(result.x),bounds,atol=1e-6))),
            correction_left_multiplied=_delta(result.x).tolist(),training_views=len(views),training_points=sum(len(p) for p,_ in observed),
            rotation_vector_rad=result.x[:3].tolist(),translation_m=result.x[3:].tolist(),calibration_applied=False)


def fit_motion_from_flow(points_anchor,anchor_pixels,target_pixels,k):
    """Fit relative camera motion from image displacement, not absolute alignment.

    Zero image displacement has identity motion even when projected anchor points
    have a fixed registration error. This prevents per-offset pose fits from
    silently replacing a shared registration. Use fit-only static landmarks and
    evaluate held-out displacement residuals separately.
    """
    p,q=_observations(points_anchor,target_pixels);anchor=np.asarray(anchor_pixels,float);k=intrinsic(k)
    if anchor.shape!=q.shape or not np.isfinite(anchor).all():raise ValueError('Finite anchor image pixels required')
    projected=project(p,np.eye(4),k);flow=q-anchor
    def residual(x):return (project(p,_delta(x),k)-projected-flow).ravel()
    bounds=np.r_[[np.deg2rad(5.)]*3,[.1]*3]
    result=least_squares(residual,np.zeros(6),bounds=(-bounds,bounds),loss='soft_l1',f_scale=1.5,max_nfev=150)
    return _delta(result.x),dict(optimizer_success=bool(result.success),bound_hit=bool(np.any(np.isclose(abs(result.x),bounds,atol=1e-6))),
            fit_median_flow_px=float(np.median(np.linalg.norm(residual(result.x).reshape(-1,2),axis=1))))


def flow_errors(points_anchor,anchor_pixels,target_pixels,k,motion):
    p,q=_observations(points_anchor,target_pixels)
    predicted=project(p,rigid(motion),k)-project(p,np.eye(4),k)
    return np.linalg.norm(predicted-(q-np.asarray(anchor_pixels)),axis=1)
