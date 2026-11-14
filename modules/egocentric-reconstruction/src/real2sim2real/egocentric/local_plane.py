"""Bounded local support hypotheses with independent, untrimmed residual checks."""

import numpy as np


def _points(value):
    p = np.asarray(value, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("At least three finite metric points required")
    return p


def fit_support_plane(points_m, *, inlier_m=0.003, max_tilt_deg=20, seed=0, trials=500):
    """RANSAC proposal in a z-up world; not a completion/accuracy certificate."""
    p = _points(points_m)
    if not np.isfinite([inlier_m, max_tilt_deg]).all() or not (
        0 < inlier_m <= 0.05
        and 0 < max_tilt_deg < 90
        and isinstance(trials, int)
        and trials > 0
    ):
        raise ValueError("Bounded positive fit parameters required")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-10:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < np.cos(np.deg2rad(max_tilt_deg)):
            continue
        mask = np.abs(p @ n - a @ n) <= inlier_m
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < 3:
        raise ValueError("No nondegenerate support-plane consensus")
    q = p[best]
    center = q.mean(axis=0)
    _, s, vt = np.linalg.svd(q - center, full_matrices=False)
    if s[1] < 1e-6:
        raise ValueError("Support points are collinear")
    n = vt[-1]
    if n[2] < 0:
        n = -n
    if n[2] < np.cos(np.deg2rad(max_tilt_deg)):
        raise ValueError("Refitted support tilt outside bound")
    return {
        "normal": n.tolist(),
        "offset_m": float(center @ n),
        "fit_points": len(p),
        "consensus_points": int(best.sum()),
        "inlier_m": inlier_m,
        "seed": seed,
    }


def assess_plane(points_m, plane, *, tolerance_m=0.003):
    """Report ALL reserved observations; never censor errors with the fit mask."""
    p = _points(points_m)
    n = np.asarray(plane["normal"], float)
    offset = float(plane["offset_m"])
    if (
        n.shape != (3,)
        or not np.isfinite(n).all()
        or not np.isfinite(offset)
        or not np.isclose(np.linalg.norm(n), 1)
        or not np.isfinite(tolerance_m)
        or tolerance_m <= 0
    ):
        raise ValueError("Unit finite normal, offset and positive tolerance required")
    e = np.abs(p @ n - offset)
    return {
        "points": len(p),
        "median_m": float(np.median(e)),
        "p90_m": float(np.quantile(e, 0.9)),
        "maximum_m": float(e.max()),
        "within_tolerance_fraction": float(np.mean(e <= tolerance_m)),
    }


def plane_patch_pose(plane, center_xy_m, thickness_m):
    """Box pose whose top face lies on the plane; finite extents are caller-owned."""
    n = np.asarray(plane["normal"], float)
    xy = np.asarray(center_xy_m, float)
    if (
        n.shape != (3,)
        or xy.shape != (2,)
        or not np.isfinite(n).all()
        or not np.isfinite(xy).all()
        or not np.isclose(np.linalg.norm(n), 1)
        or n[2] <= 0
        or not np.isfinite(thickness_m)
        or thickness_m <= 0
        or not np.isfinite(plane["offset_m"])
    ):
        raise ValueError("Finite upward unit plane, center and thickness required")
    x = np.array([1.0, 0.0, 0.0]) - n * n[0]
    x /= np.linalg.norm(x)
    top = np.r_[xy, (plane["offset_m"] - xy @ n[:2]) / n[2]]
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(n, x), n])
    pose[:3, 3] = top - n * thickness_m / 2
    return pose
