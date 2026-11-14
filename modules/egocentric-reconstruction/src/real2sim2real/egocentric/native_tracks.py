"""Native rectified stereo landmarks and fixed-anchor camera estimation.

No object/reference trajectories are inputs. A failed pose is returned as NaNs;
it cannot become a new tracking seed. Reserved landmark identities are excluded
from both RANSAC and pose refinement.
"""

import cv2
import numpy as np

from .camera import intrinsic


def stereo_landmarks(
    left,
    right,
    pixels,
    reprojection,
    *,
    maximum_roundtrip_px=0.7,
    maximum_vertical_px=1.0,
    depth_range_m=(0.2, 5.0),
):
    pixels = np.asarray(pixels, np.float32)
    Q = np.asarray(reprojection, float)
    if (
        pixels.ndim != 2
        or pixels.shape[1] != 2
        or not np.isfinite(pixels).all()
        or np.shape(left) != np.shape(right)
        or np.ndim(left) != 2
        or Q.shape != (4, 4)
        or not np.isfinite(Q).all()
        or not 0 < depth_range_m[0] < depth_range_m[1]
        or not np.isfinite([maximum_roundtrip_px, maximum_vertical_px]).all()
        or min(maximum_roundtrip_px, maximum_vertical_px) <= 0
    ):
        raise ValueError(
            "Matching grayscale images, finite pixels/Q and positive limits required"
        )
    if not len(pixels):
        return np.empty((0, 3)), np.empty(0, bool), np.empty((0, 2))
    a = pixels.reshape(-1, 1, 2)
    b, good, _ = cv2.calcOpticalFlowPyrLK(
        left,
        right,
        a,
        None,
        winSize=(25, 25),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    )
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(
        right, left, b, None, winSize=(25, 25), maxLevel=5
    )
    a, b, back = a[:, 0], b[:, 0], back[:, 0]
    disparity = a[:, 0] - b[:, 0]
    homogeneous = (
        np.c_[a, disparity, np.ones(len(a))] @ np.asarray(reprojection, float).T
    )
    xyz = homogeneous[:, :3] / np.where(
        abs(homogeneous[:, 3:]) > 1e-12, homogeneous[:, 3:], np.nan
    )
    valid = good.ravel().astype(bool) & reverse.ravel().astype(bool)
    valid &= np.linalg.norm(a - back, axis=1) <= maximum_roundtrip_px
    valid &= abs(a[:, 1] - b[:, 1]) <= maximum_vertical_px
    valid &= np.isfinite(xyz).all(1) & (disparity > 0)
    valid &= (xyz[:, 2] >= depth_range_m[0]) & (xyz[:, 2] <= depth_range_m[1])
    h, w = left.shape
    valid &= (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
    return xyz, valid, b


def fit_camera(
    points_world_m,
    pixels,
    k,
    fit_mask,
    *,
    minimum_inliers=20,
    maximum_validation_p90_px=3.0,
):
    p, q = np.asarray(points_world_m, float), np.asarray(pixels, float)
    train = np.asarray(fit_mask, bool)
    k = intrinsic(k)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or q.shape != (len(p), 2)
        or train.shape != (len(p),)
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or not isinstance(minimum_inliers, int)
        or minimum_inliers < 4
        or not np.isfinite(maximum_validation_p90_px)
        or maximum_validation_p90_px <= 0
    ):
        raise ValueError("Finite paired world/image observations and split required")
    failure = np.full((4, 4), np.nan)
    if train.sum() < minimum_inliers or (~train).sum() < 6:
        return failure, {"accepted": False, "reason": "insufficient_fit_or_validation"}
    cv2.setRNGSeed(0)
    ok, rv, tv, inliers = cv2.solvePnPRansac(
        p[train],
        q[train],
        np.asarray(k, float),
        None,
        iterationsCount=300,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < minimum_inliers:
        return failure, {"accepted": False, "reason": "pnp_consensus"}
    ids = inliers.ravel()
    rv, tv = cv2.solvePnPRefineLM(p[train][ids], q[train][ids], k, None, rv, tv)
    projected = cv2.projectPoints(p, rv, tv, k, None)[0][:, 0]
    error = np.linalg.norm(projected - q, axis=1)
    validation = error[~train]
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv.ravel()
    in_front = np.all((p @ T[:3, :3].T + T[:3, 3])[:, 2] > 0)
    passed = bool(
        in_front and np.quantile(validation, 0.9) <= maximum_validation_p90_px
    )
    report = {
        "accepted": passed,
        "fit_inliers": len(ids),
        "validation_points": len(validation),
        "validation_median_px": float(np.median(validation)),
        "validation_p90_px": float(np.quantile(validation, 0.9)),
        "fit_inlier_median_px": float(np.median(error[train][ids])),
        "reason": None if passed else "reserved_reprojection",
    }
    return (np.linalg.inv(T) if passed else failure), report
