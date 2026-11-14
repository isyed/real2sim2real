"""Fixed-identity, three-camera geometry and temporal diagnostics."""

import cv2
import numpy as np
from .camera import intrinsic, rigid


def reciprocal_indices(first, second, ratio=0.7):
    """Mutual descriptor identities, without fitting image geometry."""
    if not 0 < ratio < 1 or len(first) < 2 or len(second) < 2:
        raise ValueError("At least two descriptors and ratio in (0,1) required")
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(first, second, k=2)
    reverse = matcher.knnMatch(second, first, k=2)
    result = []
    for pair in forward:
        if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
            continue
        m = pair[0]
        back = reverse[m.trainIdx]
        if (
            len(back) == 2
            and back[0].distance < ratio * back[1].distance
            and back[0].trainIdx == m.queryIdx
        ):
            result.append((m.queryIdx, m.trainIdx))
    return np.asarray(result, int).reshape(-1, 2)


def three_camera_identities(features):
    """Close all three descriptor edges; collapse duplicate pixel orientations.

    Input is three (pixels, descriptors) tuples. Output indices address each
    camera in that same order. This is appearance support, not metric admission.
    """
    if len(features) != 3:
        raise ValueError("Three feature sets required")
    ab = reciprocal_indices(features[0][1], features[1][1])
    ac = dict(reciprocal_indices(features[0][1], features[2][1]))
    bc = dict(reciprocal_indices(features[1][1], features[2][1]))
    rows = []
    seen = set()
    for a, b in ab:
        if a not in ac or b not in bc or ac[a] != bc[b]:
            continue
        ids = [a, b, ac[a]]
        key = tuple(np.round(features[2][0][ids[2]], 2))
        if key not in seen:
            seen.add(key)
            rows.append(ids)
    return np.asarray(rows, int).reshape(-1, 3)


def stereo_to_target(left, right, target, left_pixels, right_pixels):
    """Triangulate two raw rasters and project into a third with fixed cameras.

    No RGB observations enter triangulation. Coordinates are metres in the left
    optical frame. Negative/nonfinite depths stay invalid, with NaN projections.
    """
    a, b = [np.asarray(p, float) for p in [left_pixels, right_pixels]]
    if (
        a.ndim != 2
        or a.shape[1] != 2
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not len(a)
    ):
        raise ValueError("Nonempty paired finite pixels required")
    t = np.linalg.inv(rigid(right["rig_from_camera"])) @ rigid(left["rig_from_camera"])
    u = np.linalg.inv(rigid(target["rig_from_camera"])) @ rigid(left["rig_from_camera"])
    normalized = [
        cv2.undistortPoints(
            p[:, None], intrinsic(s["intrinsics"]), np.asarray(s["distortion"])
        ).reshape(-1, 2)
        for s, p in [(left, a), (right, b)]
    ]
    h = cv2.triangulatePoints(
        np.c_[np.eye(3), np.zeros(3)], t[:3], normalized[0].T, normalized[1].T
    )
    finite = abs(h[3]) > 1e-10
    xyz = np.full((len(a), 3), np.nan)
    xyz[finite] = (h[:3, finite] / h[3, finite]).T
    right_xyz = xyz @ t[:3, :3].T + t[:3, 3]
    target_xyz = xyz @ u[:3, :3].T + u[:3, 3]
    valid = finite & (xyz[:, 2] > 0) & (right_xyz[:, 2] > 0) & (target_xyz[:, 2] > 0)
    projected = np.full_like(a, np.nan)
    if valid.any():
        projected[valid] = cv2.projectPoints(
            target_xyz[valid],
            np.zeros(3),
            np.zeros(3),
            intrinsic(target["intrinsics"]),
            np.asarray(target["distortion"]),
        )[0][:, 0]
    return xyz, projected, valid


def common_offset_errors(projected, observed, source_indices, offsets, valid):
    """Compare every offset on exactly the same identities and source frames.

    Arrays are time×identity×2; valid is image-track/metric validity before
    scoring. No error-based trimming. Missing identities remain unavailable.
    """
    p, q = np.asarray(projected, float), np.asarray(observed, float)
    valid = np.asarray(valid, bool)
    ids = np.asarray(source_indices, int)
    if (
        p.shape != q.shape
        or p.ndim != 3
        or p.shape[2] != 2
        or valid.shape != p.shape[:2]
        or not len(ids)
        or not len(offsets)
    ):
        raise ValueError("Matching time/identity grids and nonempty sweep required")
    if (
        ids.min() < 0
        or ids.max() >= len(p)
        or min(ids.min() + d for d in offsets) < 0
        or max(ids.max() + d for d in offsets) >= len(p)
    ):
        raise ValueError("Offsets exceed available frames")
    common = np.all(valid[ids] & np.isfinite(p[ids]).all(2), axis=0)
    for d in offsets:
        common &= np.all(valid[ids + d] & np.isfinite(q[ids + d]).all(2), axis=0)
    errors = np.array([q[ids + d][:, common] - p[ids][:, common] for d in offsets])
    return common, errors
