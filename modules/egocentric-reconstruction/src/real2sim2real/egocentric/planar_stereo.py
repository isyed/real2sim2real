"""Connected local plane hypotheses from calibrated stereo photometry.

Works on selected visible surface interiors. Completion, thickness and topology
are caller-declared hypotheses. Pixel agreement is not independent metric truth.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from .camera import intrinsic, rigid


def plane_points(pixels, k, world_from_camera, plane_abc):
    """Intersect rays with world z=a*x+b*y+c. Return points and optical depth."""
    uv = np.asarray(pixels, float)
    K = intrinsic(k)
    T = rigid(world_from_camera)
    abc = np.asarray(plane_abc, float)
    if (
        uv.ndim != 2
        or uv.shape[1] != 2
        or K.shape != (3, 3)
        or T.shape != (4, 4)
        or abc.shape != (3,)
        or not all(np.isfinite(x).all() for x in [uv, K, T, abc])
    ):
        raise ValueError("Finite pixels, calibration, pose and plane required")
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    directions = rays @ T[:3, :3].T
    normal = np.r_[-abc[:2], 1.0]
    denom = directions @ normal
    depth = np.divide(
        abc[2] - T[:3, 3] @ normal,
        denom,
        out=np.full(len(uv), np.nan),
        where=abs(denom) > 1e-10,
    )
    return T[:3, 3] + directions * depth[:, None], depth


def sample_pair(view, plane_abc):
    uv = np.asarray(view["pixels"], float)
    _, depth = plane_points(
        uv, view["intrinsics"], view["world_from_camera"], plane_abc
    )
    disparity = view["focal_baseline_m_px"] / depth
    right_uv = uv - np.c_[disparity, np.zeros(len(uv))]
    left_image, right_image = (
        np.asarray(view["left"], float),
        np.asarray(view["right"], float),
    )
    if (
        left_image.ndim != 2
        or left_image.shape != right_image.shape
        or not len(uv)
        or not np.isfinite(depth).all()
        or (depth <= 0).any()
        or not np.isfinite(view["focal_baseline_m_px"])
        or view["focal_baseline_m_px"] <= 0
    ):
        raise ValueError(
            "Matching grayscale images and positive visible stereo depth required"
        )
    h, w = left_image.shape
    for pixels in (uv, right_uv):
        if (
            (pixels < 0).any()
            or (pixels[:, 0] > w - 1).any()
            or (pixels[:, 1] > h - 1).any()
        ):
            raise ValueError(
                "Stereo projection outside image; do not clamp evidence to border"
            )
    left = map_coordinates(
        np.asarray(view["left"], float), uv[:, ::-1].T, order=1, mode="nearest"
    )
    right = map_coordinates(
        np.asarray(view["right"], float), right_uv[:, ::-1].T, order=1, mode="nearest"
    )
    return left, right, disparity


def fit_plane_photometry(
    views, initial_abc, *, slope_bound=0.3, height_bounds_m=(0, 0.04)
):
    if len(views) < 2:
        raise ValueError("Multiple fit views required")
    initial = np.asarray(initial_abc, float)
    if (
        initial.shape != (3,)
        or not np.isfinite(initial).all()
        or not np.isfinite(slope_bound)
        or slope_bound <= 0
        or len(height_bounds_m) != 2
        or not np.isfinite(height_bounds_m).all()
        or height_bounds_m[0] >= height_bounds_m[1]
    ):
        raise ValueError("Finite initial plane required")

    def residual(abc):
        errors = []
        for view in views:
            left, right, _ = sample_pair(view, abc)
            # Per-view brightness offset is a nuisance, never a geometric warp.
            diff = right - left
            diff -= np.median(diff)
            errors.append(diff / 8 / np.sqrt(len(diff)))
        return np.concatenate(errors)

    candidates = []
    lo = [-slope_bound, -slope_bound, height_bounds_m[0]]
    hi = [slope_bound, slope_bound, height_bounds_m[1]]
    for dc in [-0.006, 0, 0.006]:
        x = np.clip(initial + [0, 0, dc], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        fit = least_squares(
            residual,
            x,
            bounds=(lo, hi),
            loss="soft_l1",
            f_scale=0.1,
            diff_step=1e-4,
            max_nfev=100,
        )
        candidates.append(fit)
    fit = min(candidates, key=lambda x: x.cost)
    return fit.x, {
        "optimizer_success": bool(fit.success),
        "cost": float(fit.cost),
        "candidates": [
            {"abc": f.x.tolist(), "cost": float(f.cost)} for f in candidates
        ],
    }


def photometric_check(view, plane_abc):
    a, b, d = sample_pair(view, plane_abc)
    error = b - a
    offset = float(np.median(error))
    error -= offset
    corr = float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-8 else None
    return {
        "pixels": len(a),
        "brightness_offset": offset,
        "median_gray_error": float(np.median(abs(error))),
        "p90_gray_error": float(np.quantile(abs(error), 0.9)),
        "correlation": corr,
        "minimum_disparity_px": float(np.min(d)),
        "maximum_disparity_px": float(np.max(d)),
    }
