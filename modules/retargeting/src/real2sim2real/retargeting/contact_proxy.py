"""Local solid-span comparison against the union of convex collision pieces."""

import numpy as np


def merge_intervals(intervals, tolerance=1e-8):
    merged = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([float(lo), float(hi)])
    return merged


def convex_interval(mesh, origin, axis):
    """Clip a line to outward half-spaces of a convex, wound closed mesh."""
    normals = mesh.face_normals
    offsets = np.einsum("ij,ij->i", normals, np.asarray(origin) - mesh.triangles[:, 0])
    slope = normals @ axis
    parallel = abs(slope) < 1e-10
    if (offsets[parallel] > 1e-8).any():
        return None
    upper = -offsets[slope > 1e-10] / slope[slope > 1e-10]
    lower = -offsets[slope < -1e-10] / slope[slope < -1e-10]
    if not len(upper) or not len(lower):
        raise ValueError("Closed bounded convex pieces required")
    lo, hi = float(lower.max()), float(upper.min())
    return [lo, hi] if hi > lo else None


def local_span_audit(
    source_mesh,
    convex_pieces,
    center,
    axis,
    *,
    max_boundary_error_m=0.001,
    max_relative_error=0.25,
):
    center, axis = np.asarray(center, float), np.asarray(axis, float)
    if (
        center.shape != (3,)
        or axis.shape != (3,)
        or not np.isfinite([center, axis]).all()
        or not np.isclose(np.linalg.norm(axis), 1)
        or not source_mesh.is_watertight
        or not source_mesh.is_winding_consistent
        or not convex_pieces
        or not np.isfinite([max_boundary_error_m, max_relative_error]).all()
        or max_boundary_error_m <= 0
        or not 0 < max_relative_error < 1
    ):
        raise ValueError("Closed source and finite local span audit required")
    if any(
        not m.is_watertight
        or not m.is_winding_consistent
        or not m.is_convex
        or m.volume <= 0
        for m in convex_pieces
    ):
        raise ValueError("Collision pieces must be closed outward convex solids")
    extent = float(np.linalg.norm(source_mesh.extents)) + 0.1
    points, _, faces = source_mesh.ray.intersects_location(
        [center - axis * extent], [axis], multiple_hits=True
    )
    points = np.asarray(points).reshape(-1, 3)
    order = np.argsort((points - center) @ axis)
    points, faces = points[order], faces[order]
    depths = (points - center) @ axis
    signs = source_mesh.face_normals[faces] @ axis
    source_intervals = [
        [float(a), float(b)]
        for a, b, na, nb in zip(depths[:-1], depths[1:], signs[:-1], signs[1:])
        if na < -0.05 and nb > 0.05
    ]
    proxy_intervals = merge_intervals(
        [
            i
            for m in convex_pieces
            if (i := convex_interval(m, center, axis)) is not None
        ]
    )
    source = next((i for i in source_intervals if i[0] <= 0 <= i[1]), None)
    proxy = next((i for i in proxy_intervals if i[0] <= 0 <= i[1]), None)
    row = dict(
        source_intervals_m=source_intervals,
        proxy_union_intervals_m=proxy_intervals,
        source_width_m=None,
        proxy_width_m=None,
        boundary_error_m=None,
        accepted=False,
        reason="missing_local_solid_span",
    )
    if source is not None and proxy is not None:
        width = source[1] - source[0]
        error = float(np.max(abs(np.asarray(source) - proxy)))
        limit = min(max_boundary_error_m, max_relative_error * width)
        row.update(
            source_width_m=width,
            proxy_width_m=proxy[1] - proxy[0],
            boundary_error_m=error,
            boundary_limit_m=limit,
            accepted=error <= limit,
            reason="accepted" if error <= limit else "local_proxy_boundary_error",
        )
    return row
