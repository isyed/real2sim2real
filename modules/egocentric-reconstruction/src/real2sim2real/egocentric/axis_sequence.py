"""Observation-fitted axisymmetric motion with an explicit source-clock prior.

Temporal regularization produces hypotheses, not additional observations. The
caller must evaluate reserved geometry, endpoints and motion before admission.
No controller states or evaluation reference poses are inputs.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .axis_motion import axis_frame
from .camera import intrinsic, rigid
from .tapered_container import outer_distance


def fit_axis_sequence(
    points_camera,
    endpoint_pixels,
    cameras,
    k,
    dimensions,
    timestamps_s,
    initial_poses,
    *,
    endpoint_sigma_px=15.0,
    surface_sigma_m=0.003,
    acceleration_scale_m_s2=3.0,
    axis_acceleration_scale_s2=50.0,
    max_evaluations=100,
):
    """Fit fixed-shape centres and directed axes using measured time intervals.

    Point lists contain FIT observations only. Empty lists are allowed but must
    remain explicitly unobserved in the caller's result. Axial yaw is a gauge.
    The smoothness scales are modeling priors, not measured uncertainty bounds.
    """
    t = np.asarray(timestamps_s, float)
    n = len(t)
    dims = np.asarray(dimensions, float)
    cams = np.asarray([rigid(x) for x in cameras])
    K = intrinsic(k)
    seed = np.asarray([rigid(x) for x in initial_poses])
    uv = np.asarray(endpoint_pixels, float)
    scales = np.array(
        [
            endpoint_sigma_px,
            surface_sigma_m,
            acceleration_scale_m_s2,
            axis_acceleration_scale_s2,
        ]
    )
    if (
        n < 3
        or cams.shape != (n, 4, 4)
        or seed.shape != (n, 4, 4)
        or uv.shape != (n, 2, 2)
        or not np.isfinite(uv).all()
        or not np.isfinite(t).all()
        or (np.diff(t) <= 0).any()
        or dims.shape != (3,)
        or not np.isfinite(dims).all()
        or (dims <= 0).any()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
        or len(points_camera) != n
    ):
        raise ValueError(
            "Finite observations, positive scales and increasing source timestamps required"
        )
    points = [np.asarray(p, float) for p in points_camera]
    if any(p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() for p in points):
        raise ValueError("Finite metric point lists required")
    x0 = np.c_[seed[:, :3, 3] + seed[:, :3, 2] * dims[2] / 2, seed[:, :3, 2]]
    rows = sum(len(p) + 5 for p in points) + 6 * (n - 2)
    sparsity = lil_matrix((rows, n * 6), dtype=int)
    offset = 0
    for i, p in enumerate(points):
        count = len(p) + 5
        sparsity[offset : offset + count, i * 6 : (i + 1) * 6] = 1
        offset += count
    for i in range(1, n - 1):
        sparsity[offset : offset + 6, (i - 1) * 6 : (i + 2) * 6] = 1
        offset += 6

    def unpack(flat):
        x = flat.reshape(n, 6)
        norm = np.linalg.norm(x[:, 3:], axis=1)
        axis = x[:, 3:] / np.maximum(norm[:, None], 1e-8)
        return x[:, :3], axis, norm

    def residual(flat):
        center, axis, norm = unpack(flat)
        out = []
        for i, p in enumerate(points):
            bottom = center[i] - axis[i] * dims[2] / 2
            # Signed axial coordinate and radial distance avoid arbitrary spin.
            world = p @ cams[i, :3, :3].T + cams[i, :3, 3]
            delta = world - bottom
            z = delta @ axis[i]
            radial = np.linalg.norm(delta - z[:, None] * axis[i], axis=1)
            local = np.c_[radial, np.zeros(len(p)), z]
            out.append(
                outer_distance(local, dims) / surface_sigma_m / np.sqrt(max(1, len(p)))
            )
            ends = np.array([bottom, bottom + axis[i] * dims[2]])
            camera = (ends - cams[i, :3, 3]) @ cams[i, :3, :3]
            q = camera @ K.T
            out.append(
                (
                    (q[:, :2] / np.maximum(q[:, 2:], 0.01) - uv[i]) / endpoint_sigma_px
                ).ravel()
            )
            out.append(np.array([(norm[i] - 1) * 10]))
        velocity = np.diff(center, axis=0) / np.diff(t)[:, None]
        axis_velocity = np.diff(axis, axis=0) / np.diff(t)[:, None]
        dt = (t[2:] - t[:-2]) / 2
        acceleration = np.diff(velocity, axis=0) / dt[:, None] / acceleration_scale_m_s2
        axis_acceleration = (
            np.diff(axis_velocity, axis=0) / dt[:, None] / axis_acceleration_scale_s2
        )
        out.append(np.c_[acceleration, axis_acceleration].ravel())
        return np.concatenate(out)

    fit = least_squares(
        residual,
        x0.ravel(),
        jac_sparsity=sparsity.tocsr(),
        loss="linear",
        max_nfev=max_evaluations,
        ftol=1e-5,
    )
    center, axis, _ = unpack(fit.x)
    poses = []
    gauge = None
    for c, z in zip(center, axis):
        R = axis_frame(z, gauge)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c - z * dims[2] / 2
        poses.append(T)
        gauge = R
    return np.array(poses), {
        "optimizer_success": bool(fit.success),
        "evaluations": fit.nfev,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "endpoint_sigma_px": endpoint_sigma_px,
        "surface_sigma_m": surface_sigma_m,
        "acceleration_scale_m_s2": acceleration_scale_m_s2,
        "axis_acceleration_scale_s2": axis_acceleration_scale_s2,
        "axial_spin_observed": False,
        "temporal_regularization_is_a_prior": True,
        "time_warped": False,
    }
