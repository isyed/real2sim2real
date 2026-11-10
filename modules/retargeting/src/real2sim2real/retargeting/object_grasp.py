"""Object-surface grasp proposals independent of human finger predictions."""

import numpy as np


def cue_cost(center, cues, *, max_weight=0.02):
    """Optional uncertain spatial cues affect ranking, never feasibility gates."""
    total = 0.0
    if not np.isfinite(max_weight) or max_weight < 0:
        raise ValueError("Nonnegative finite cue weight required")
    for cue in cues:
        point = np.asarray(cue["position_m"], float)
        confidence = float(cue["confidence"])
        weight = float(cue.get("weight", max_weight))
        if (
            point.shape != (3,)
            or not np.isfinite(point).all()
            or not 0 <= confidence <= 1
            or not np.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("Finite metric cue and confidence in [0,1] required")
        total += (
            min(max_weight, weight)
            * confidence
            * np.linalg.norm(np.asarray(center) - point)
        )
    return float(total)


def surface_pairs(
    mesh,
    *,
    min_width_m,
    max_width_m,
    minimum_height_m,
    max_candidates=12,
    cues=(),
    finger_offsets_m=(-0.012, 0.012),
):
    """Sample opposing outer-surface rays along principal horizontal directions.

    The narrow-span search can find handles without assigning semantic labels.
    Antipodality is a geometric proposal filter, not force-closure certification.
    """
    if (
        not 0 < min_width_m < max_width_m
        or type(max_candidates) is not int
        or not 1 <= max_candidates <= 100
    ):
        raise ValueError("Bounded positive grasp width and candidate budget required")
    vertices = np.asarray(mesh.vertices)
    offsets = np.asarray(finger_offsets_m, float)
    if (
        offsets.ndim != 1
        or not 1 <= len(offsets) <= 5
        or not np.isfinite(offsets).all()
        or (np.abs(offsets) > 0.1).any()
        or not np.isfinite(minimum_height_m)
    ):
        raise ValueError("Finite bounded finger offsets and support height required")
    if minimum_height_m >= mesh.bounds[1, 2] - 0.005:
        return []
    _, axes = np.linalg.eigh(np.cov(vertices[:, :2].T))
    center = mesh.bounds.mean(0)
    proposals = []
    for index in range(2):
        close = np.r_[axes[:, index], 0.0]
        along = np.cross([0.0, 0.0, 1.0], close)
        t = (vertices - center) @ along
        extent = float(np.ptp(vertices @ close)) + 0.1
        origins = np.asarray(
            [
                center + along * y - close * extent + [0, 0, z - center[2]]
                for y in np.linspace(t.min() + 0.008, t.max() - 0.008, 40)
                for z in np.linspace(
                    max(minimum_height_m, mesh.bounds[0, 2] + 0.005),
                    mesh.bounds[1, 2] - 0.005,
                    9,
                )
            ]
        )
        locations, rays, faces = mesh.ray.intersects_location(
            origins, np.tile(close, (len(origins), 1)), multiple_hits=True
        )
        for ray in np.unique(rays):
            ids = np.flatnonzero(rays == ray)
            ids = ids[np.argsort(locations[ids] @ close)]
            if len(ids) < 2:
                continue
            lo, hi = ids[[0, -1]]
            width = float((locations[hi] - locations[lo]) @ close)
            if not min_width_m <= width <= max_width_m:
                continue
            normals = mesh.face_normals[faces[[lo, hi]]]
            if -normals[0] @ close < 0.5 or normals[1] @ close < 0.5:
                continue
            midpoint = (locations[lo] + locations[hi]) / 2
            # Adjacent finger contacts must also lie on the represented surface.
            targets = [locations[lo]]
            good = True
            for offset in finger_offsets_m:
                o = midpoint + along * offset - close * extent
                p, _, _ = mesh.ray.intersects_location(
                    o[None], close[None], multiple_hits=True
                )
                if len(p) < 2:
                    good = False
                    break
                p = p[np.argmax(p @ close)]
                if np.linalg.norm(p - midpoint) > max_width_m:
                    good = False
                    break
                targets.append(p)
            if not good:
                continue
            score = (
                abs(width - 0.035)
                + 0.15 * np.linalg.norm(midpoint - mesh.center_mass)
                + cue_cost(midpoint, cues)
            )
            proposals.append(
                dict(
                    center_m=midpoint.tolist(),
                    width_m=width,
                    close_axis=close.tolist(),
                    along_axis=along.tolist(),
                    targets_m=np.asarray(targets).tolist(),
                    score=float(score),
                )
            )
    selected = []
    for row in sorted(proposals, key=lambda p: p["score"]):
        if any(
            np.linalg.norm(np.asarray(row["center_m"]) - p["center_m"]) < 0.012
            for p in selected
        ):
            continue
        selected.append(row)
        if len(selected) == max_candidates:
            break
    return selected
