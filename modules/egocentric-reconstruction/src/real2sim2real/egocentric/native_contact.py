"""Contact-local diagnostics for frozen tapered geometry in native stereo."""

import numpy as np


def side_coordinates(points, dimensions):
    p = np.asarray(points, float)
    rb, rt, h = np.asarray(dimensions, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or min(rb, rt, h) <= 0
    ):
        raise ValueError("Finite points and positive frozen dimensions required")
    radius = np.linalg.norm(p[:, :2], axis=1)
    expected = rb + (rt - rb) * p[:, 2] / h
    return p[:, 2] / h, (radius - expected) / np.sqrt(1 + ((rt - rb) / h) ** 2)


def tangent_evidence(points, center, normal, axial, span, *, cells=4):
    """Report finite patch coverage and plane agreement without promoting a grasp.

    Search has a 20 mm normal window; that is a correspondence neighborhood, not
    an allowed contact gap. Empty cells and missing observations remain explicit.
    Rectangle bounds conservatively enclose the supplied actual-pad footprint.
    """
    p = np.asarray(points, float)
    c = np.asarray(center, float)
    n = np.asarray(normal, float)
    a = np.asarray(axial, float)
    span = np.asarray(span, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or c.shape != (3,)
        or n.shape != (3,)
        or a.shape != (3,)
        or span.shape != (2,)
        or not np.isfinite(np.r_[c, n, a, span]).all()
        or (span <= 0).any()
        or cells < 1
    ):
        raise ValueError("Finite geometry and positive footprint required")
    if (
        not np.isclose(np.linalg.norm(n), 1)
        or not np.isclose(np.linalg.norm(a), 1)
        or abs(n @ a) > 1e-6
    ):
        raise ValueError("Orthogonal unit tangent and normal required")
    b = np.cross(n, a)
    q = (p - c) @ np.column_stack([a, b, n])
    keep = (abs(q[:, :2]) <= span / 2).all(1) & (abs(q[:, 2]) <= 0.02)
    q = q[keep]
    if not len(q):
        return dict(
            points=0,
            coverage_fraction=0.0,
            plane_p90_m=None,
            normal_error_deg=None,
            normal_offset_m=None,
        )
    bins = np.clip(np.floor((q[:, :2] / span + 0.5) * cells).astype(int), 0, cells - 1)
    result = dict(
        points=len(q),
        coverage_fraction=len(np.unique(bins, axis=0)) / (cells * cells),
        normal_offset_m=float(np.median(q[:, 2])),
        plane_p90_m=None,
        normal_error_deg=None,
    )
    if len(q) >= 3:
        centered = q - np.mean(q, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pn = vt[-1]
        result.update(
            plane_p90_m=float(np.quantile(abs(centered @ pn), 0.9)),
            normal_error_deg=float(np.rad2deg(np.arccos(np.clip(abs(pn[2]), 0, 1)))),
        )
    return result


def place_distal_sweep(
    vertices, pad_point, body_basis, center, normal, axial, distances
):
    """Place actual distal geometry at a proposed pad, then withdraw radially.

    This is a necessary geometry screen, not an articulated hand insertion path.
    No joints are solved; proximal links, palm and wrist are absent.
    """
    v = np.asarray(vertices, float)
    n = -np.asarray(normal, float)
    a = np.asarray(axial, float)
    basis = np.asarray(body_basis, float)
    d = np.asarray(distances, float)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or basis.shape != (3, 3)
        or d.ndim != 1
        or not len(d)
        or not np.isfinite(np.r_[v.ravel(), basis.ravel(), d]).all()
        or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    ):
        raise ValueError("Finite vertices, rotation basis and distances required")
    world = np.column_stack([a, np.cross(n, a), n])
    rotation = world @ basis.T
    if not np.allclose(world.T @ world, np.eye(3), atol=1e-6):
        raise ValueError("Orthogonal unit contact frame required")
    points = (v - np.asarray(pad_point)) @ rotation.T + np.asarray(center)
    return points[None] + d[:, None, None] * np.asarray(normal)
