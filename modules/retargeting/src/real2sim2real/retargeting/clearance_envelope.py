"""Conservative whole-object translation bounds for pre-rotation clearance."""

import numpy as np


def vertical_clearance_target(
    points_about_center_m,
    obstacle_upper_z_m,
    *,
    clearance_m,
    tilt_reserve_rad=0.0,
    tracking_reserve_m=0.0,
):
    """Required center height above obstacles for a bounded orientation change.

    Points are convex-geometry vertices relative to the planning center, expressed
    in the current world orientation. A rotation of at most theta moves any point
    by at most 2*r*sin(theta/2). The bound therefore includes the entire geometry,
    not just its center. Obstacle scope is explicit: callers must select all
    potentially intersected material, including uncertain cells. This is an
    endpoint envelope, not a collision-checked insertion path or arm IK result.
    Reserves are design margins, never a change to collision acceptance limits.
    """
    points = np.asarray(points_about_center_m, float)
    values = [obstacle_upper_z_m, clearance_m, tilt_reserve_rad, tracking_reserve_m]
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(values).all()
        or clearance_m < 0
        or tracking_reserve_m < 0
        or not 0 <= tilt_reserve_rad <= np.pi
    ):
        raise ValueError(
            "Finite metric geometry and nonnegative bounded reserves required"
        )
    radius = np.linalg.norm(points, axis=1)
    displacement = 2 * radius * np.sin(tilt_reserve_rad / 2)
    lower = float(np.min(points[:, 2] - displacement))
    return {
        "required_center_z_m": float(
            obstacle_upper_z_m + clearance_m + tracking_reserve_m - lower
        ),
        "lowest_relative_z_bound_m": lower,
        "nominal_lowest_relative_z_m": float(points[:, 2].min()),
        "maximum_radius_m": float(radius.max()),
        "obstacle_upper_z_m": float(obstacle_upper_z_m),
        "clearance_m": float(clearance_m),
        "tilt_reserve_rad": float(tilt_reserve_rad),
        "tracking_reserve_m": float(tracking_reserve_m),
        "endpoint_bound_only": True,
        "path_admitted": False,
    }
