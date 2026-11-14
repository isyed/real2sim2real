"""Explicit contact chirality and bounded articulated insertion diagnostics."""

import numpy as np
from .conditional_hand_scene import sample_paths


def opposed_taper_targets(
    center,
    dimensions,
    height_fraction,
    azimuth_rad,
    spread_rad,
    finger_height_offset_m=0.0,
    chirality=1,
    *,
    layout_roll_rad=0.0,
):
    """Thumb and two opposing fingers on a tapered exterior, in explicit order.

    Chirality mirrors the index/middle angular ordering; it does not flip pad
    normals, joint signs or the robot model. Dimensions are bottom/rim radius
    and height in metres. Layout roll trades angular finger separation for
    vertical separation on the wall; it is not a commanded wrist rotation.
    Zero preserves the original layout. This is a proposal, not observed contact.
    """
    center = np.asarray(center, float)
    dims = np.asarray(dimensions, float)
    values = np.asarray(
        [
            height_fraction,
            azimuth_rad,
            spread_rad,
            finger_height_offset_m,
            layout_roll_rad,
        ],
        float,
    )
    if (
        center.shape != (3,)
        or dims.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(dims).all()
        or (dims <= 0).any()
        or not np.isfinite(values).all()
        or chirality not in [-1, 1]
        or not 0 < height_fraction < 1
        or not 0 < spread_rad < np.pi / 2
    ):
        raise ValueError(
            "Finite tapered dimensions and explicit mirrored layout required"
        )
    rb, rt, h = dims
    vertical_spread = (
        (rb + (rt - rb) * height_fraction)
        * np.sin(spread_rad)
        * np.sin(layout_roll_rad)
    )
    z = np.array(
        [
            h * height_fraction,
            h * height_fraction + finger_height_offset_m - vertical_spread,
            h * height_fraction + finger_height_offset_m + vertical_spread,
        ]
    )
    if (z <= 0).any() or (z >= h).any():
        raise ValueError("Contact heights must lie on the exterior wall")
    angles = np.array(
        [
            azimuth_rad,
            azimuth_rad + np.pi - chirality * spread_rad * np.cos(layout_roll_rad),
            azimuth_rad + np.pi + chirality * spread_rad * np.cos(layout_roll_rad),
        ]
    )
    radial = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])
    p = (
        center
        + radial * (rb + (rt - rb) * z / h)[:, None]
        + np.column_stack([np.zeros(3), np.zeros(3), z])
    )
    normals = radial + [0, 0, -(rt - rb) / h]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return p, normals


def check_insertion_candidates(
    model, data, indices, candidates, *, max_penetration_m=0.003
):
    """Audit every declared joint-space path without changing caller state.

    Each candidate is a complete articulated path; no endpoint-only acceptance.
    All enabled hand geoms are checked. A finite candidate budget prevents an
    accidentally unbounded search. Full obstacle restoration remains mandatory.
    """
    if not 0 < max_penetration_m <= 0.01 or not 1 <= len(candidates) <= 64:
        raise ValueError("Bounded path count and penetration limit required")
    results = []
    for name, waypoints in candidates.items():
        audit, samples = sample_paths(model, data, indices, waypoints)
        results.append(
            dict(
                name=name,
                audit=audit,
                samples=samples,
                passed=bool(
                    audit["within_joint_limits"]
                    and audit["maximum_penetration_m"] <= max_penetration_m
                ),
            )
        )
    return results


def insertion_waypoints(
    preshape, opened, closed, slide_indices, retreat_m, *, shape_during_insert=False
):
    """Build four explicit joint waypoints; caller chooses every finger posture.

    Preserves unused fingers exactly as supplied. Translation is in the declared
    fixture-slide coordinates, not an asserted Cartesian collision-free path.
    Always pass the result through full-hand collision checks.
    """
    pre, op, cl = [np.asarray(q, float) for q in [preshape, opened, closed]]
    slides = np.asarray(slide_indices)
    retreat = np.asarray(retreat_m, float)
    if (
        pre.ndim != 1
        or not len(pre)
        or any(q.shape != pre.shape or not np.isfinite(q).all() for q in [pre, op, cl])
        or slides.shape != (3,)
        or slides.dtype.kind not in "iu"
        or len(set(slides.tolist())) != 3
        or (slides < 0).any()
        or (slides >= len(pre)).any()
        or retreat.shape != (3,)
        or not np.isfinite(retreat).all()
        or not 0 < np.linalg.norm(retreat) <= 0.25
    ):
        raise ValueError(
            "Finite matching postures, three distinct slides and bounded retreat required"
        )
    far = pre.copy()
    far[slides] += retreat
    middle = pre.copy()
    if shape_during_insert:
        middle = (pre + op) / 2
        middle[slides] += 0.5 * retreat
    return np.array([far, middle, op, cl])
