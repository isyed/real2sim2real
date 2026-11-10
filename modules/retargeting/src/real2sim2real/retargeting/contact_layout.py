"""Bounded local surface spans and gravity lever diagnostics for grasp planning.

Local entry/exit pairs include rim walls missed by outermost-surface rays.
These are geometric proposals, not frictional force-closure certificates.
"""

import numpy as np

from .object_grasp import cue_cost


def gravity_lever(center_m, center_of_mass_m, gravity=(0.0, 0.0, -9.81)):
    """Moment arm and required balancing torque per kg about a contact centre."""
    center, com, g = [
        np.asarray(v, float) for v in (center_m, center_of_mass_m, gravity)
    ]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (center, com, g)):
        raise ValueError("Finite three-dimensional points and gravity required")
    norm = np.linalg.norm(g)
    if norm <= 0:
        raise ValueError("Nonzero gravity required")
    torque = np.cross(com - center, g)
    return dict(
        gravity_lever_m=float(np.linalg.norm(torque) / norm),
        gravity_torque_per_kg_nm=torque.tolist(),
        center_of_mass_m=com.tolist(),
    )


def _hits(mesh, origin, axis):
    points, _, faces = mesh.ray.intersects_location(
        np.asarray(origin)[None], np.asarray(axis)[None], multiple_hits=True
    )
    if not len(points):
        return np.empty((0, 3)), np.empty((0, 3))
    order = np.argsort(points @ axis)
    return points[order], mesh.face_normals[faces[order]]


def contact_targets(mesh, pair, offsets, *, direction=1):
    """Place thumb on the selected side; fingers on the opposite local surface.

    Offsets are ordered anatomical finger locations along ``pair.along_axis``.
    Nearest surface selection avoids jumping across a cavity to its far wall.
    Missing or wrong-facing surfaces reject the proposal rather than fabricating
    contacts. Returns None when any requested finger lacks a suitable surface.
    """
    if direction not in (-1, 1):
        raise ValueError("Direction must be +1 or -1")
    center = np.asarray(pair["center_m"], float)
    axis, along = [np.asarray(pair[k], float) for k in ("close_axis", "along_axis")]
    offsets = np.asarray(offsets, float)
    if (
        any(v.shape != (3,) or not np.isfinite(v).all() for v in (center, axis, along))
        or not np.allclose(
            [axis @ axis, along @ along, axis @ along], [1, 1, 0], atol=1e-6
        )
        or offsets.ndim != 1
        or not 1 <= len(offsets) <= 5
        or not np.isfinite(offsets).all()
        or (np.abs(offsets) > 0.1).any()
        or not np.isfinite(pair["width_m"])
        or pair["width_m"] <= 0
    ):
        raise ValueError(
            "Finite contact frame, width and bounded finger offsets required"
        )
    # Older outer-ray proposals do not store endpoints; recover the central pair.
    extent = float(np.linalg.norm(mesh.extents)) + 0.1
    if "endpoints_m" not in pair:
        p, _ = _hits(mesh, center - axis * extent, axis)
        if len(p) < 2:
            return None
        anchors = p[[0, -1]]
    else:
        anchors = np.asarray(pair["endpoints_m"], float)
        if anchors.shape != (2, 3) or not np.isfinite(anchors).all():
            raise ValueError("Two finite surface endpoints required")
    thumb, opposing = (
        (anchors[0], anchors[1]) if direction == 1 else (anchors[1], anchors[0])
    )
    targets = [thumb]
    for offset in offsets:
        expected = opposing + along * offset
        p, normals = _hits(mesh, expected - axis * extent, axis)
        facing = normals @ (axis * direction) > 0.5
        p = p[facing]
        if not len(p):
            return None
        nearest = p[np.argmin(np.linalg.norm(p - expected, axis=1))]
        # A local wall must continue nearby; crossing the pot is not a finger span.
        if np.linalg.norm(nearest - expected) > min(0.02, pair["width_m"]):
            return None
        if np.linalg.norm(nearest - center) > pair.get("max_width_m", 0.1):
            return None
        targets.append(nearest)
    return np.asarray(targets)


def surface_layouts(
    mesh,
    *,
    min_width_m,
    max_width_m,
    minimum_height_m,
    max_candidates=12,
    finger_offsets_m=(-0.03, -0.01, 0.01, 0.03),
    cues=(),
    yaw_samples=8,
    gravity_lever_weight=1.0,
    diagnostics=None,
):
    """Rank local solid spans over bounded horizontal directions near the COM.

    Only consecutive entering/exiting intersections of a consistently oriented,
    closed mesh are paired: empty space inside a vessel is never a solid span.
    Both thumb-side layouts must have surface-supported opposing-finger targets.
    Uniform-density mesh COM is an assumption, not an observed mass distribution.
    """
    offsets = np.asarray(finger_offsets_m, float)
    if (
        not np.isfinite(
            [min_width_m, max_width_m, minimum_height_m, gravity_lever_weight]
        ).all()
        or not 0 < min_width_m < max_width_m
        or gravity_lever_weight < 0
        or type(max_candidates) is not int
        or not 1 <= max_candidates <= 100
        or type(yaw_samples) is not int
        or not 2 <= yaw_samples <= 16
        or offsets.ndim != 1
        or not 1 <= len(offsets) <= 5
        or not np.isfinite(offsets).all()
        or (np.abs(offsets) > 0.1).any()
    ):
        raise ValueError("Finite bounded layout search parameters required")
    if not mesh.is_volume:
        raise ValueError("Local spans require a closed, consistently oriented volume")
    low = max(minimum_height_m, mesh.bounds[0, 2] + 0.005)
    high = mesh.bounds[1, 2] - 0.005
    if low >= high:
        return []
    vertices = np.asarray(mesh.vertices)
    _, axes = np.linalg.eigh(np.cov(vertices[:, :2].T))
    angle0 = np.arctan2(axes[1, 0], axes[0, 0])
    center = mesh.bounds.mean(0)
    extent = float(np.linalg.norm(mesh.extents)) + 0.1
    proposals = []
    solid_widths, solid_levers = [], []
    for angle in angle0 + np.arange(yaw_samples) * np.pi / yaw_samples:
        close = np.array([np.cos(angle), np.sin(angle), 0])
        along = np.cross([0, 0, 1], close)
        projected = (vertices - center) @ along
        origins = np.array(
            [
                center + along * y - close * extent + [0, 0, z - center[2]]
                for y in np.linspace(
                    projected.min() + 0.008, projected.max() - 0.008, 40
                )
                for z in np.linspace(low, high, 9)
            ]
        )
        points, rays, faces = mesh.ray.intersects_location(
            origins, np.tile(close, (len(origins), 1)), multiple_hits=True
        )
        for ray in np.unique(rays):
            ids = np.flatnonzero(rays == ray)
            ids = ids[np.argsort(points[ids] @ close)]
            for lo, hi in zip(ids[:-1], ids[1:]):
                width = float((points[hi] - points[lo]) @ close)
                normals = mesh.face_normals[faces[[lo, hi]]] @ close
                if not (normals[0] < -0.5 and normals[1] > 0.5):
                    continue
                midpoint = (points[lo] + points[hi]) / 2
                lever = gravity_lever(midpoint, mesh.center_mass)
                solid_widths.append(width)
                solid_levers.append(lever["gravity_lever_m"])
                if not min_width_m <= width <= max_width_m:
                    continue
                pair = dict(
                    center_m=midpoint.tolist(),
                    width_m=width,
                    close_axis=close.tolist(),
                    along_axis=along.tolist(),
                    endpoints_m=points[[lo, hi]].tolist(),
                    max_width_m=max_width_m,
                    targets_m=[points[lo].tolist()],
                    **lever,
                    score=float(
                        abs(width - 0.035)
                        + gravity_lever_weight * lever["gravity_lever_m"]
                        + cue_cost(midpoint, cues)
                    ),
                )
                proposals.append(pair)
    selected = []
    missing_targets = 0
    for pair in sorted(proposals, key=lambda p: p["score"]):
        if any(
            np.linalg.norm(np.asarray(pair["center_m"]) - p["center_m"]) < 0.012
            and abs(np.dot(pair["close_axis"], p["close_axis"])) > 0.95
            for p in selected
        ):
            continue
        forward = contact_targets(mesh, pair, offsets)
        reverse = contact_targets(mesh, pair, offsets, direction=-1)
        if forward is None or reverse is None:
            missing_targets += 1
            continue
        pair["targets_m"] = forward.tolist()
        pair["reverse_targets_m"] = reverse.tolist()
        selected.append(pair)
        if len(selected) == max_candidates:
            break
    if diagnostics is not None:
        diagnostics.update(
            sampled_solid_spans=len(solid_widths),
            width_admitted_spans=len(proposals),
            examined_missing_finger_surface=missing_targets,
            selected_pairs=len(selected),
            candidate_budget_reached=len(selected) == max_candidates,
            solid_span_width_m=solid_widths,
            solid_span_gravity_lever_m=solid_levers,
            center_of_mass_assumption="uniform density over reconstructed volume",
            scope="sampled horizontal rays only; no global feasibility certificate",
        )
    return selected
