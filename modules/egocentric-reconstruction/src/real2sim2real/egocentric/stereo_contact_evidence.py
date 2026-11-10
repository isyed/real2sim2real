"""Sparse rectified stereo checks independent of predicted contact depth.

The source image location is model-seeded, so these are consistency diagnostics,
not independent anatomical/surface annotations. Search uses a declared full
disparity range, never a window around the reconstruction's predicted depth.
"""

import cv2
import numpy as np


def _search(
    source,
    target,
    uv,
    *,
    direction,
    radius,
    disparities,
    vertical_radius,
    minimum_std,
    minimum_correlation,
    minimum_uniqueness,
):
    x, y = np.rint(uv).astype(int)
    h, w = source.shape
    if not (radius <= x < w - radius and radius <= y < h - radius):
        return dict(accepted=False, reason="outside_image")
    template = source[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(
        np.float32
    )
    std = float(template.std())
    if std < minimum_std:
        return dict(accepted=False, reason="low_texture", texture_std=std)
    scores, pixels = [], []
    for disparity in disparities:
        tx = x + direction * disparity
        candidates = []
        for dy in range(-vertical_radius, vertical_radius + 1):
            ty = y + dy
            if radius <= tx < w - radius and radius <= ty < h - radius:
                patch = target[
                    ty - radius : ty + radius + 1, tx - radius : tx + radius + 1
                ].astype(np.float32)
                score = (
                    float(
                        cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0]
                    )
                    if patch.std() >= minimum_std
                    else -1.0
                )
                candidates.append((score, tx, ty))
        score, tx, ty = max(candidates, default=(-1.0, -1, -1))
        scores.append(score)
        pixels.append([tx, ty])
    best = int(np.argmax(scores))
    d = disparities[best]
    competing = [s for i, s in enumerate(scores) if abs(disparities[i] - d) > 2]
    margin = scores[best] - max(competing, default=-1.0)
    reason = "matched"
    if best in (0, len(scores) - 1):
        reason = "search_boundary"
    elif scores[best] < minimum_correlation:
        reason = "poor_correlation"
    elif margin < minimum_uniqueness:
        reason = "ambiguous_match"
    delta = 0.0
    if reason == "matched":
        a, b, c = scores[best - 1 : best + 2]
        denominator = a - 2 * b + c
        if denominator < -1e-8:
            delta = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
    pixel = [float(pixels[best][0] + direction * delta), float(pixels[best][1])]
    return dict(
        accepted=reason == "matched",
        reason=reason,
        source_pixel=[int(x), int(y)],
        target_pixel=pixel,
        disparity_px=float(d + delta),
        correlation=scores[best],
        uniqueness_margin=margin,
        texture_std=std,
    )


def match_rectified(
    left,
    right,
    uv,
    *,
    radius=3,
    minimum_disparity=1,
    maximum_disparity=160,
    vertical_radius=2,
    minimum_std=5.0,
    minimum_correlation=0.85,
    minimum_uniqueness=0.08,
    maximum_roundtrip_px=1.5,
):
    """NCC search with ambiguity and reverse-match gates; missing stays missing."""
    left, right, uv = np.asarray(left), np.asarray(right), np.asarray(uv)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or uv.shape != (2,)
        or not np.isfinite(uv).all()
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or any(
            type(v) is not int
            for v in [radius, minimum_disparity, maximum_disparity, vertical_radius]
        )
        or not 1 <= radius <= 8
        or not 1 <= minimum_disparity < maximum_disparity <= 512
        or maximum_disparity - minimum_disparity < 6
        or not 0 <= vertical_radius <= 3
        or not np.isfinite(
            [minimum_std, minimum_correlation, minimum_uniqueness, maximum_roundtrip_px]
        ).all()
        or minimum_std <= 0
        or not 0 < minimum_correlation <= 1
        or not 0 < minimum_uniqueness <= 2
        or maximum_roundtrip_px <= 0
    ):
        raise ValueError(
            "Finite rectified grayscale images and bounded search required"
        )
    options = dict(
        radius=radius,
        disparities=list(range(minimum_disparity, maximum_disparity + 1)),
        vertical_radius=vertical_radius,
        minimum_std=minimum_std,
        minimum_correlation=minimum_correlation,
        minimum_uniqueness=minimum_uniqueness,
    )
    result = _search(left, right, uv, direction=-1, **options)
    if not result["accepted"]:
        return result
    reverse = _search(right, left, result["target_pixel"], direction=1, **options)
    result["reverse"] = reverse
    if not reverse["accepted"]:
        result.update(accepted=False, reason="reverse_" + reverse["reason"])
    else:
        error = float(
            np.linalg.norm(np.asarray(reverse["target_pixel"]) - result["source_pixel"])
        )
        result["roundtrip_px"] = error
        if error > maximum_roundtrip_px:
            result.update(accepted=False, reason="roundtrip_error")
    return result


def triangulate_match(match, projections):
    """Return rectified-left coordinates, using observed pixels in both views."""
    if not match["accepted"]:
        raise ValueError("Rejected matches cannot provide metric evidence")
    p, q = [np.asarray(v, float) for v in projections]
    if p.shape != (3, 4) or q.shape != (3, 4) or not np.isfinite([p, q]).all():
        raise ValueError("Finite stereo projection matrices required")
    point = cv2.triangulatePoints(
        p,
        q,
        np.asarray(match["source_pixel"], float)[:, None],
        np.asarray(match["target_pixel"], float)[:, None],
    )[:, 0]
    if abs(point[3]) < 1e-10:
        raise ValueError("Stereo point at infinity")
    point = point[:3] / point[3]
    if not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError("Invalid stereo triangulation")
    return point
