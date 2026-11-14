"""Finite pad samples against an explicitly hypothetical tapered exterior."""

import numpy as np
from scipy.spatial import ConvexHull


def convex_pad_samples(vertices, point, normal, *, radius_m=0.004):
    """Ray-project a central cross onto the actual convex collision pad.

    Uses hull planes, avoiding optional spatial-index dependencies. Missing ray
    intersections stay missing; samples do not establish finite physical contact.
    """
    vertices = np.asarray(vertices, float)
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or p.shape != (3,)
        or n.shape != (3,)
        or not np.isfinite(p).all()
        or not np.isfinite(n).all()
        or np.linalg.norm(n) < 1e-9
        or not 0 < radius_m <= 0.005
    ):
        raise ValueError(
            "Finite convex pad geometry and bounded sample radius required"
        )
    n = n / np.linalg.norm(n)
    a = np.cross(n, np.eye(3)[np.argmin(abs(n))])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    planes = ConvexHull(vertices).equations
    points = []
    normals = []
    distance = float(np.linalg.norm(np.ptp(vertices, axis=0)) + 0.01)
    for offset in [
        np.zeros(3),
        radius_m * a,
        -radius_m * a,
        radius_m * b,
        -radius_m * b,
    ]:
        origin = p + offset + distance * n
        direction = -n
        den = planes[:, :3] @ direction
        value = planes[:, :3] @ origin + planes[:, 3]
        if ((abs(den) < 1e-12) & (value > 1e-9)).any():
            continue
        enter = np.flatnonzero(den < -1e-12)
        leave = np.flatnonzero(den > 1e-12)
        if not len(enter):
            continue
        times = -value[enter] / den[enter]
        selected = int(enter[np.argmax(times)])
        lo = float(times.max())
        hi = float(np.min(-value[leave] / den[leave])) if len(leave) else np.inf
        if lo < 0 or lo > hi + 1e-10:
            continue
        points.append(origin + lo * direction)
        normals.append(planes[selected, :3])
    return np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3)


def tapered_sample_evidence(
    points,
    pad_normals,
    center,
    dimensions,
    *,
    max_gap_m=0.0005,
    max_penetration_m=0.003,
    max_normal_error_deg=25.0,
):
    """Signed side-wall gap and opposing-normal tests, not force/retention proof.

    Distances refer to the ideal tapered lateral surface in the declared metric
    frame. End caps and hollow interior are excluded; height is checked explicitly.
    """
    p = np.asarray(points, float)
    n = np.asarray(pad_normals, float)
    c = np.asarray(center, float)
    dims = np.asarray(dimensions, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or n.shape != p.shape
        or c.shape != (3,)
        or dims.shape != (3,)
        or not all(np.isfinite(x).all() for x in [p, n, c, dims])
        or (dims <= 0).any()
        or not np.allclose(np.linalg.norm(n, axis=1), 1)
        or not 0 <= max_gap_m <= 0.001
        or not 0 < max_penetration_m <= 0.01
        or not 0 < max_normal_error_deg <= 90
    ):
        raise ValueError(
            "Metric surface samples, unit normals and bounded contact limits required"
        )
    local = p - c
    rho = np.linalg.norm(local[:, :2], axis=1)
    rb, rt, h = dims
    slope = (rt - rb) / h
    gaps = (rho - (rb + slope * local[:, 2])) / np.sqrt(1 + slope * slope)
    outward = np.column_stack(
        [local[:, :2] / np.maximum(rho[:, None], 1e-12), np.full(len(p), -slope)]
    )
    outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-12)
    angle = np.rad2deg(np.arccos(np.clip((-outward * n).sum(1), -1, 1)))
    side = (local[:, 2] >= 0) & (local[:, 2] <= h) & (rho > 1e-9)
    supported = (
        side
        & (gaps <= max_gap_m)
        & (gaps >= -max_penetration_m)
        & (angle <= max_normal_error_deg)
    )
    return dict(
        signed_gap_m=gaps.tolist(),
        normal_error_deg=angle.tolist(),
        supported=supported.tolist(),
        supported_samples=int(supported.sum()),
        sample_count=len(p),
        physical_contact_validated=False,
    )


def closure_stop_decision(
    support_counts,
    penetration_m,
    within_joint_limits,
    *,
    required_pads=3,
    max_penetration_m=0.003,
    frame_normal_errors_deg=None,
    max_frame_normal_error_deg=25.0,
):
    """Safety rejection precedes contact stop for a sampled kinematic diagnostic.

    Counts must already encode side-wall proximity and opposing normals. This
    decision does not establish force support or physical contact observability.
    When supplied, each pad-frame error is checked before support can stop
    closure. Call on every sample: an acceptable endpoint cannot repair an
    earlier orientation violation. Omission preserves the historical rule.
    """
    counts = np.asarray(support_counts)
    if (
        type(required_pads) is not int
        or required_pads < 1
        or not 0 < max_penetration_m <= 0.01
        or not 0 < max_frame_normal_error_deg <= 90
    ):
        raise ValueError("Explicit pad count and bounded penetration limit required")
    if (
        counts.shape != (required_pads,)
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
        or not np.isfinite(penetration_m)
        or penetration_m < 0
    ):
        return "abort_invalid_observation"
    if within_joint_limits is not True or penetration_m > max_penetration_m:
        return "abort_geometry"
    if frame_normal_errors_deg is not None:
        angles = np.asarray(frame_normal_errors_deg, float)
        if (
            angles.shape != (required_pads,)
            or not np.isfinite(angles).all()
            or (angles < 0).any()
            or (angles > 180).any()
        ):
            return "abort_invalid_observation"
        if (angles > max_frame_normal_error_deg).any():
            return "abort_orientation"
    return "stop_on_support" if (counts > 0).all() else "continue"
