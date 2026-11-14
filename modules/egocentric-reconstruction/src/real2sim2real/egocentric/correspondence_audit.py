"""Identity and spatial diagnostics that never optimize camera parameters."""

import numpy as np


def error_summary(errors):
    """An empty cohort is unavailable, never a zero-error success."""
    e = np.asarray(errors, float)
    if e.ndim != 1 or not np.isfinite(e).all():
        raise ValueError("Finite scalar residuals required")
    return dict(
        count=len(e),
        median_px=float(np.median(abs(e))) if len(e) else None,
        p90_px=float(np.quantile(abs(e), 0.9)) if len(e) else None,
        signed_median_px=float(np.median(e)) if len(e) else None,
    )


def spatial_groups(pixels, size, principal_point, *, columns=4, rows=3):
    """Raw raster bins and distance from principal point / half raster diagonal."""
    p = np.asarray(pixels, float)
    size = np.asarray(size, float)
    principal_point = np.asarray(principal_point, float)
    if principal_point.shape != (2,) or not np.isfinite(principal_point).all():
        raise ValueError("Finite principal point required")
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or not np.isfinite(p).all()
        or size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or min(columns, rows) < 1
    ):
        raise ValueError(
            "Finite pixel pairs and positive raster/grid dimensions required"
        )
    if np.any(p < 0) or np.any(p >= size):
        raise ValueError("Pixels outside raster")
    cell = np.floor(p / size * [columns, rows]).astype(int)
    radius = np.linalg.norm(p - np.asarray(principal_point), axis=1) / (
        np.linalg.norm(size) / 2
    )
    return cell[:, 1] * columns + cell[:, 0], radius


def cycle_consistency(
    source,
    target,
    source_middle,
    middle,
    middle_target,
    target_end,
    *,
    identity_tolerance_px=1.0,
):
    """Join A→B and B→C evidence to saved A→C matches by raw pixel identity.

    Missing or spatially ambiguous joins remain unavailable. No nearest-neighbor
    extrapolation outside the tolerance, and no epipolar-error based selection.
    """
    arrays = [
        np.asarray(p, float)
        for p in (source, target, source_middle, middle, middle_target, target_end)
    ]
    if any(p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() for p in arrays):
        raise ValueError("Finite pixel arrays required")
    a, c, ab, b, bc, end = arrays
    if (
        len(a) != len(c)
        or len(ab) != len(b)
        or len(bc) != len(end)
        or identity_tolerance_px <= 0
    ):
        raise ValueError("Paired observations and positive identity tolerance required")
    errors = np.full(len(a), np.nan)
    for i, p in enumerate(a):
        first = np.flatnonzero(np.linalg.norm(ab - p, axis=1) <= identity_tolerance_px)
        if not len(first):
            continue
        candidates = []
        for j in first:
            second = np.flatnonzero(
                np.linalg.norm(bc - b[j], axis=1) <= identity_tolerance_px
            )
            candidates.extend(end[second])
        if not candidates:
            continue
        candidates = np.asarray(candidates)
        # Duplicate SIFT orientations at the same point are not distinct evidence.
        if (
            np.max(np.linalg.norm(candidates[:, None] - candidates[None], axis=2))
            > identity_tolerance_px
        ):
            continue
        errors[i] = np.linalg.norm(candidates.mean(axis=0) - c[i])
    return errors


def image_features(image, mask=None):
    """Raw SIFT identities; optional raster mask focuses detection without cropping."""
    import cv2

    keypoints, descriptors = cv2.SIFT_create(nfeatures=4000).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask
    )
    if descriptors is None or len(descriptors) < 2:
        raise ValueError("Insufficient image texture for reciprocal matching")
    return np.array([k.pt for k in keypoints]), descriptors


def reciprocal_support(a, b, fa, fb):
    """Bidirectional ratio-0.7 descriptor support for saved pixel identities."""
    import cv2

    pa, da = fa
    pb, db = fb
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(da, db, k=2)
    reverse = matcher.knnMatch(db, da, k=2)
    good = []
    for ms in forward:
        if len(ms) != 2 or ms[0].distance >= 0.7 * ms[1].distance:
            continue
        m = ms[0]
        back = reverse[m.trainIdx]
        if (
            len(back) == 2
            and back[0].distance < 0.7 * back[1].distance
            and back[0].trainIdx == m.queryIdx
        ):
            good.append((m.queryIdx, m.trainIdx))
    accepted = np.zeros(len(a), bool)
    for ia, ib in good:
        accepted |= (np.linalg.norm(a - pa[ia], axis=1) < 0.01) & (
            np.linalg.norm(b - pb[ib], axis=1) < 0.01
        )
    return accepted
