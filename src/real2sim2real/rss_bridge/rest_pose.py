"""Bounded single-pose fitting for a user-verified stationary RGB interval.

The caller supplies rendered silhouettes (for example RSS's native renderer).
This cannot determine when an object is stationary; require source evidence.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from .bundle import rigid


def blend_stationary_tail(poses, rest_pose, *, start, fully_resting):
    """Explicit pose interpolation during a measured settling interval.

    This is an inferred repair, not a recovered observation. Validate every
    changed frame and both temporal seams independently before promotion.
    """
    original = rigid(poses)
    rest = rigid(rest_pose)
    if (original.ndim != 3 or rest.shape != (4, 4)
            or not 0 <= start < fully_resting < len(original)):
        raise ValueError("Complete trajectory and ordered in-range transition required")
    result = original.copy()
    for i in range(start+1, fully_resting):
        u = (i-start)/(fully_resting-start)
        w = u*u*(3-2*u)
        result[i, :3, :3] = Slerp([0, 1], Rotation.from_matrix(
            np.stack([original[i, :3, :3], rest[:3, :3]])))([w]).as_matrix()[0]
        result[i, :3, 3] = (1-w)*original[i, :3, 3] + w*rest[:3, 3]
    result[fully_resting:] = rest
    return result


def fit_pose(render_mask, seed_pose, vertices, masks, *, max_rotation_deg=15.,
             center_shift_bounds_m=(.02, .02, .06), max_evaluations=180,
             max_seconds=300., support_plane=None, support_tolerance_m=.02,
             preserve_center_depth=False):
    seed = rigid(seed_pose)
    vertices = np.asarray(vertices, float)
    masks = np.asarray(masks)
    bounds = np.asarray(center_shift_bounds_m, float)
    if (seed.shape != (4, 4) or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or masks.ndim != 3
            or masks.dtype != bool or len(masks) < 2
            or np.any(masks.sum(axis=(1, 2)) < 100) or bounds.shape != (3,)
            or not np.isfinite(bounds).all() or np.any(bounds <= 0)
            or not 0 < max_rotation_deg <= 25 or max_evaluations < 1 or max_seconds <= 0
            or type(preserve_center_depth) is not bool):
        raise ValueError("Explicit bounded pose and multiple observed masks required")
    plane = None if support_plane is None else np.asarray(support_plane, float)
    if plane is not None and (plane.shape != (4,) or not np.isfinite(plane).all()
                              or not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-5)):
        raise ValueError("Support plane must have a unit normal")
    center = vertices.mean(0)
    center_camera = seed[:3, :3] @ center + seed[:3, 3]
    angle = np.deg2rad(max_rotation_deg)/np.sqrt(3)
    dimension = 5 if preserve_center_depth else 6
    def pose(x):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_rotvec(x[:3]*angle).as_matrix() @ seed[:3, :3]
        shift = np.r_[x[3:]*bounds[:2], 0.] if preserve_center_depth else x[3:]*bounds
        result[:3, 3] = center_camera + shift - result[:3, :3] @ center
        return result
    started = time.monotonic()
    records = []
    best = [np.inf, seed.copy(), np.zeros(dimension)]
    def objective(x):
        if len(records) >= max_evaluations or time.monotonic()-started > max_seconds:
            raise TimeoutError("Bounded RSS fit reached its limit")
        p = pose(x)
        rendered = np.asarray(render_mask(p))
        if rendered.dtype != bool or rendered.shape != masks.shape[1:]:
            raise ValueError("Renderer must return a boolean mask at the native ROI resolution")
        intersection = np.count_nonzero(masks & rendered, axis=(1, 2))
        union = np.count_nonzero(masks | rendered, axis=(1, 2))
        overlaps = intersection / np.maximum(1, union)
        penetration = 0.
        if plane is not None:
            world = vertices @ p[:3, :3].T+p[:3, 3]
            penetration = max(0., -float(np.min(world @ plane[:3] + plane[3])))
        value = float(1-overlaps.mean()+.001*np.mean(x*x)
                      + 10*max(0., penetration-support_tolerance_m))
        records.append(dict(evaluation=len(records)+1, loss=value,
                            mean_iou=float(overlaps.mean()), support_penetration_m=penetration))
        if value < best[0]:
            best[:] = [value, p, x.copy()]
        return value
    converged = False
    reason = "optimizer_finished"
    try:
        objective(np.zeros(dimension))
        result = minimize(objective, np.zeros(dimension), method="Powell",
                          bounds=[(-1., 1.)]*dimension,
                          options=dict(maxfev=max_evaluations, maxiter=8, xtol=.02, ftol=1e-5))
        converged = bool(result.success)
        reason = str(result.message)
    except TimeoutError as error:
        reason = str(error)
    return best[1], dict(evaluations=len(records), converged=converged, stop_reason=reason,
                         seconds=time.monotonic()-started, records=records,
                         max_rotation_deg=max_rotation_deg,
                         center_shift_bounds_m=bounds.tolist(),
                         camera_center_depth_preserved=preserve_center_depth,
                         best_normalized_parameters=best[2].tolist(),
                         proposal_only=True, source_validated=False, physics_validated=False)
