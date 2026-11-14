"""Static multi-view object anchors with explicit support assumptions."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import intrinsic, rigid
from .container_axis import endpoints
from .tapered_container import outer_distance


def fit_anchor(
    views,
    dimensions_m,
    initial_world_from_object,
    *,
    upright_on_table_z_m=None,
    surface_sigma_m=0.003,
    endpoint_sigma_px=15.0,
):
    """Fit one pose across FIT views only; caller reserves complete other views.

    Optional upright/table constraint is a declared source interpretation, not
    a measured force contact. No shape or camera parameter is optimized.
    """
    initial = rigid(initial_world_from_object)
    dims = np.asarray(dimensions_m, float)
    if len(views) < 2 or min(surface_sigma_m, endpoint_sigma_px) <= 0:
        raise ValueError("Multiple views and positive residual scales required")
    fixed = upright_on_table_z_m is not None
    if fixed and not np.isfinite(upright_on_table_z_m):
        raise ValueError("Finite table height required")
    rows = []
    for v in views:
        points = np.asarray(v["points_camera_m"], float)
        uv = np.asarray(v["endpoint_pixels"], float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 3
            or not np.isfinite(points).all()
            or uv.shape != (2, 2)
            or not np.isfinite(uv).all()
        ):
            raise ValueError(
                "Finite metric points and two ordered endpoint cues required"
            )
        visible = np.asarray(v.get("endpoint_visible", [True, True]), bool)
        if visible.shape != (2,):
            raise ValueError("Two endpoint visibility flags required")
        rows.append(
            (
                points,
                uv,
                rigid(v["world_from_camera"]),
                intrinsic(v["intrinsics"]),
                visible,
            )
        )

    def pose(x):
        T = np.eye(4)
        if fixed:
            T[:3, 3] = [x[0], x[1], upright_on_table_z_m]
        else:
            T[:3, 3] = x[:3]
            T[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix()
        return T

    def residual(x):
        T = pose(x)
        out = []
        for p, uv, C, K, visible in rows:
            native = np.linalg.inv(C) @ T
            out.append(
                outer_distance((p - native[:3, 3]) @ native[:3, :3], dims)
                / surface_sigma_m
                / np.sqrt(len(p))
            )
            out.append(
                (
                    (endpoints(native, dims[2], K) - uv)[visible] / endpoint_sigma_px
                ).ravel()
            )
        return np.concatenate(out)

    x0 = (
        initial[:2, 3]
        if fixed
        else np.r_[initial[:3, 3], Rotation.from_matrix(initial[:3, :3]).as_rotvec()]
    )
    fit = least_squares(residual, x0, max_nfev=250, loss="soft_l1")
    return pose(fit.x), {
        "optimizer_success": bool(fit.success),
        "fit_cost": float(fit.cost),
        "fit_views": len(views),
        "shape_changed": False,
        "camera_changed": False,
        "axial_spin_observed": False,
        "upright_table_assumption": fixed,
        "table_z_m": upright_on_table_z_m,
        "static_pose_assumption": True,
        "control_ready": False,
    }
