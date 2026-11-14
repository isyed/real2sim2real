"""Explicit 180-degree pixel and optical-axis convention hypotheses."""

import copy
import cv2
import numpy as np
from .camera import intrinsic, rigid

TURN = np.diag([-1.0, -1.0, 1.0, 1.0])


def turn_pixel_calibration(k, distortion, size):
    """Positive-focal OpenCV calibration for a 180-degree raster rotation.

    Optical coordinates change by diag(-1,-1,1). Radial terms are unchanged;
    tangential, thin-prism and tilt terms reverse sign. Extrinsic conversion is
    deliberately separate because supplied metadata may already transform axes.
    """
    k = intrinsic(k).copy()
    d = np.array(distortion, float, copy=True)
    if d.ndim != 1 or len(d) not in [4, 5, 8, 12, 14] or not np.isfinite(d).all():
        raise ValueError("Supported finite OpenCV perspective distortion required")
    width, height = size
    if min(width, height) <= 0:
        raise ValueError("Positive raster dimensions required")
    k[0, 2] = width - 1 - k[0, 2]
    k[1, 2] = height - 1 - k[1, 2]
    d[2:4] *= -1
    if len(d) >= 12:
        d[8:12] *= -1
    if len(d) == 14:
        d[12:14] *= -1
    return k, d


def convention_streams(streams, *, pixels=False, axes=False):
    result = copy.deepcopy(streams)
    for s in result.values():
        if pixels:
            k, d = turn_pixel_calibration(s["intrinsics"], s["distortion"], s["size"])
            s["intrinsics"], s["distortion"] = k.tolist(), d.tolist()
        if axes:
            s["rig_from_camera"] = (TURN @ rigid(s["rig_from_camera"]) @ TURN).tolist()
    return result


def pair_residuals(source, target, source_pixels, target_pixels):
    """No pose fit: calibrated Sampson error and signed triangulated depths.

    A reversed translation can satisfy epipolar geometry while triangulating
    everything behind the cameras. Always report cheirality alongside error.
    """
    k1, k2 = [intrinsic(s["intrinsics"]) for s in [source, target]]
    t = np.linalg.inv(rigid(target["rig_from_camera"])) @ rigid(
        source["rig_from_camera"]
    )
    observed = []
    normalized = []
    for s, k, p in zip([source, target], [k1, k2], [source_pixels, target_pixels]):
        p = np.asarray(p, float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("Finite matched pixels required")
        n = cv2.undistortPoints(p[:, None], k, np.asarray(s["distortion"])).reshape(
            -1, 2
        )
        normalized.append(n)
        h = np.c_[n, np.ones(len(n))] @ k.T
        observed.append(h)
    if len(observed[0]) != len(observed[1]) or len(observed[0]) < 8:
        raise ValueError("At least eight paired observations required")
    x, y, z = t[:3, 3]
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    f = np.linalg.inv(k2).T @ cross @ t[:3, :3] @ np.linalg.inv(k1)
    a, b = observed
    la = a @ f.T
    lb = b @ f
    signed = np.sum(b * la, axis=1) / np.maximum(
        np.sqrt((la[:, :2] ** 2).sum(1) + (lb[:, :2] ** 2).sum(1)), 1e-12
    )
    h = cv2.triangulatePoints(
        np.c_[np.eye(3), np.zeros(3)], t[:3], normalized[0].T, normalized[1].T
    )
    finite = abs(h[3]) > 1e-10
    p = np.full((len(a), 3), np.nan)
    p[finite] = (h[:3, finite] / h[3, finite]).T
    q = p @ t[:3, :3].T + t[:3, 3]
    positive = finite & (p[:, 2] > 0) & (q[:, 2] > 0)
    return dict(
        signed_sampson_px=signed,
        positive_depth=positive,
        source_depth_m=p[:, 2],
        baseline_m=float(np.linalg.norm(t[:3, 3])),
    )


def pair_geometry(source, target, source_pixels, target_pixels):
    """Summarize calibrated errors without discarding any match identities."""
    residuals = pair_residuals(source, target, source_pixels, target_pixels)
    error = abs(residuals["signed_sampson_px"])
    positive = residuals["positive_depth"]
    return dict(
        matches=len(error),
        sampson_median_px=float(np.median(error)),
        sampson_p90_px=float(np.quantile(error, 0.9)),
        positive_depth_fraction=float(positive.mean()),
        baseline_m=residuals["baseline_m"],
        source_depth_q10_q50_q90_m=np.quantile(
            residuals["source_depth_m"][positive], [0.1, 0.5, 0.9]
        ).tolist()
        if positive.any()
        else None,
        transform_fitted=False,
    )
