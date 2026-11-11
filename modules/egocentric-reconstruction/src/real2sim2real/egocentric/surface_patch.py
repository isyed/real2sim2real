"""Conservative finite-patch diagnostics from observed metric stereo points.

The model supplies the region and a normal sign, never missing observations.
Image correspondences remain correlated; a planar fit is not a thickness or
insertion-clearance measurement.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def fit_patch(
    points_m,
    center_m,
    normal_hint,
    *,
    minimum_points=9,
    radius_m=0.012,
    pad_radius_m=0.004,
    maximum_p90_residual_m=0.001,
    maximum_normal_error_deg=15,
):
    points, center, hint = map(
        lambda x: np.asarray(x, float), (points_m, center_m, normal_hint)
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or center.shape != (3,)
        or hint.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(center).all()
        or not np.isfinite(hint).all()
        or not np.isclose(np.linalg.norm(hint), 1)
        or type(minimum_points) is not int
        or not 6 <= minimum_points <= 100
        or not 0 < pad_radius_m < radius_m <= 0.03
        or not 0 < maximum_p90_residual_m <= 0.005
        or not 0 < maximum_normal_error_deg <= 45
    ):
        raise ValueError(
            "Finite metric observations and explicit bounded patch limits required"
        )
    points = np.unique(points, axis=0)
    points = points[np.linalg.norm(points - center, axis=1) <= radius_m]
    result = dict(
        points_in_region=len(points),
        accepted=False,
        finite_bearing_area_verified=False,
        thickness_verified=False,
        insertion_clearance_verified=False,
    )
    if len(points) < minimum_points:
        return dict(result, rejected_for=["insufficient_metric_points"])
    centroid = points.mean(0)
    _, singular, axes = np.linalg.svd(points - centroid, full_matrices=False)
    normal = axes[2] * (1 if axes[2] @ hint >= 0 else -1)
    xy = (points - centroid) @ axes[:2].T
    target = (center - centroid) @ axes[:2].T
    residual = np.abs((points - centroid) @ normal)
    angle = float(np.rad2deg(np.arccos(np.clip(normal @ hint, -1, 1))))
    try:
        hull = ConvexHull(xy)
        # SciPy hull equations have unit outward normals in the plane.
        margin = float(np.min(-(hull.equations[:, :2] @ target + hull.equations[:, 2])))
        area = float(hull.volume)
    except QhullError:
        margin, area = 0.0, 0.0
    p90 = float(np.percentile(residual, 90))
    rejected = []
    if singular[1] < 1e-8 or margin < pad_radius_m:
        rejected.append("insufficient_two_dimensional_coverage")
    if p90 > maximum_p90_residual_m:
        rejected.append("nonplanar_or_noisy_depth")
    if angle > maximum_normal_error_deg:
        rejected.append("normal_disagrees_with_proposal")
    return dict(
        result,
        accepted=not rejected,
        rejected_for=rejected,
        centroid_m=centroid.tolist(),
        normal=normal.tolist(),
        normal_error_deg=angle,
        p90_residual_m=p90,
        center_plane_distance_m=float(abs((center - centroid) @ normal)),
        observed_hull_area_m2=area,
        center_in_hull_margin_m=margin,
        span_m=np.ptp(xy, axis=0).tolist(),
        coverage_is_sampled_convex_hull=True,
    )


def normal_repeatability(fits, *, minimum_frames=3, maximum_spread_deg=10):
    if (
        type(minimum_frames) is not int
        or not 2 <= minimum_frames <= 40
        or not np.isfinite(maximum_spread_deg)
        or not 0 < maximum_spread_deg <= 45
    ):
        raise ValueError("Explicit bounded repeated-view limits required")
    normals = np.array([f["normal"] for f in fits if f.get("accepted") is True])
    if not len(normals):
        return dict(accepted=False, accepted_frames=0, maximum_spread_deg=None)
    if (
        normals.ndim != 2
        or normals.shape[1] != 3
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1)
    ):
        raise ValueError("Finite oriented unit normals required")
    spread = float(np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    return dict(
        accepted=len(normals) >= minimum_frames and spread <= maximum_spread_deg,
        accepted_frames=len(normals),
        maximum_spread_deg=spread,
    )
