"""Visibility proxies and temporal measurements on observed metric hands.

Neither low visibility nor a neighboring pose creates an observation. Temporal
statistics use adjacent, accepted samples and the actual capture timestamps.
"""

import numpy as np


def visibility_weights(keypoints, joints, depth, mask, object_mask, *, radius=4):
    """Fixed image support proxy; depth reads only the fitting checkerboard.

    Bones lie inside skin, so depth disagreement is only a coarse occlusion
    cue. These weights are not calibrated probabilities of joint visibility.
    """
    keypoints, joints = np.asarray(keypoints), np.asarray(joints)
    if keypoints.shape != (21, 2) or joints.shape != (21, 3):
        raise ValueError("Require 21 image and camera-space joints")
    if not np.isfinite(keypoints).all() or not np.isfinite(joints).all():
        raise ValueError("Visibility requires a measured, finite hand")
    if depth.shape != mask.shape or mask.shape != object_mask.shape or mask.ndim != 2:
        raise ValueError("Depth and masks must share image dimensions")
    weights, labels = np.full(21, 0.2), []
    height, width = mask.shape
    for j, (u, v) in enumerate(np.rint(keypoints).astype(int)):
        if not (0 <= u < width and 0 <= v < height):
            labels.append("outside_image")
            continue
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        z = depth[y0:y1, x0:x1]
        fit = ((xx // 3 + yy // 3) % 2) == 0
        valid = fit & np.isfinite(z) & (z > 0.15) & (z < 1.5)
        hand = mask[y0:y1, x0:x1].astype(bool)
        occluder = object_mask[y0:y1, x0:x1].astype(bool)
        if not hand.any():
            labels.append("object_occluded" if occluder.any() else "outside_hand_mask")
        elif (valid & hand).sum() < 3:
            weights[j] = 0.5
            labels.append("hand_mask_without_fit_depth")
        elif abs(float(np.median(z[valid & hand])) - joints[j, 2]) > 0.04:
            weights[j] = 0.35
            labels.append("depth_disagreement")
        else:
            weights[j] = 1.0
            labels.append("mask_and_fit_depth_supported")
    return weights, labels


def temporal_measurements(joints, accepted, times, *, max_step_s=0.1):
    joints, accepted, times = (
        np.asarray(joints),
        np.asarray(accepted, bool),
        np.asarray(times),
    )
    if joints.shape != (*accepted.shape, 21, 3) or accepted.ndim != 2:
        raise ValueError("Require NxH hand validity and NxHx21x3 joints")
    if times.shape != (len(accepted),) or not np.isfinite(times).all():
        raise ValueError("Matching finite capture timestamps required")
    dt = np.diff(times)
    if not np.all(dt > 0) or not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("Increasing timestamps and positive maximum step required")
    if not np.isfinite(joints[accepted]).all():
        raise ValueError("Accepted joints must be finite")
    edge = accepted[1:] & accepted[:-1] & (dt <= max_step_s)[:, None]
    velocity = np.diff(joints, axis=0) / dt[:, None, None, None]
    triple = edge[1:] & edge[:-1]
    acceleration = (
        2 * np.diff(velocity, axis=0) / (dt[1:] + dt[:-1])[:, None, None, None]
    )
    return velocity, acceleration, edge, triple


def neighbor_target(joints, accepted, times, i, h, *, max_step_s=0.1):
    """A fitting prior only, never an interpolated output or missing-hand fill."""
    if not 0 < i < len(times) - 1 or not np.asarray(accepted)[i - 1 : i + 2, h].all():
        return None
    dt = np.diff(times[i - 1 : i + 2])
    if not np.isfinite(dt).all() or not ((dt > 0) & (dt <= max_step_s)).all():
        return None
    target = (dt[1] * joints[i - 1, h] + dt[0] * joints[i + 1, h]) / dt.sum()
    return target if np.isfinite(target).all() else None


def component_admission(before, proposed, accepted, selected, times, *, max_step_s=0.1):
    """Apply changes atomically per connected set of affected triplets.

    Reject a whole component if it increases acceleration or speed tails. Failed
    proposals fall back to the parent; coverage is never reduced by smoothing.
    """
    selected = np.asarray(selected, bool)
    if selected.shape != np.shape(accepted) or np.any(selected & ~accepted):
        raise ValueError("Temporal changes require accepted parent observations")
    if not np.array_equal(
        np.asarray(before)[~selected], np.asarray(proposed)[~selected], equal_nan=True
    ):
        raise ValueError("Unselected temporal anchors must remain unchanged")
    v0, a0, edges, triples = temporal_measurements(
        before, accepted, times, max_step_s=max_step_s
    )
    v1, a1, _, _ = temporal_measurements(
        proposed, accepted, times, max_step_s=max_step_s
    )
    admitted = np.zeros_like(selected)
    records = []
    for h in range(selected.shape[1]):
        ids = np.flatnonzero(selected[:, h])
        for group in np.split(ids, np.flatnonzero(np.diff(ids) > 2) + 1):
            if not len(group):
                continue
            affected = np.zeros(len(accepted), bool)
            affected[group] = True
            ts = triples[:, h] & (affected[:-2] | affected[1:-1] | affected[2:])
            es = edges[:, h] & (affected[:-1] | affected[1:])
            if not ts.any():
                records.append(
                    dict(
                        indices=group.tolist(),
                        hand=h,
                        accepted=False,
                        reasons=["no_observed_triplets"],
                    )
                )
                continue
            acceleration = [np.linalg.norm(a[:, h][ts], axis=-1) for a in (a0, a1)]
            speed = [np.linalg.norm(v[:, h][es], axis=-1) for v in (v0, v1)]
            rms0, rms1 = [float(np.sqrt(np.mean(a * a))) for a in acceleration]
            max0, max1 = [float(a.max()) for a in acceleration]
            p90v0, p90v1 = [float(np.quantile(v, 0.9)) for v in speed]
            maxv0, maxv1 = [float(v.max()) for v in speed]
            reasons = []
            if rms1 > 0.95 * rms0:
                reasons.append("insufficient_acceleration_improvement")
            if max1 > 1.05 * max0 + 1e-6:
                reasons.append("peak_acceleration_regression")
            if p90v1 > p90v0 + 0.05 or maxv1 > maxv0 + 0.1:
                reasons.append("speed_regression")
            admitted[group, h] = not reasons
            records.append(
                dict(
                    indices=group.tolist(),
                    hand=h,
                    accepted=not reasons,
                    reasons=reasons,
                    acceleration_rms_before_m_s2=rms0,
                    acceleration_rms_after_m_s2=rms1,
                    acceleration_max_before_m_s2=max0,
                    acceleration_max_after_m_s2=max1,
                    speed_p90_before_m_s=p90v0,
                    speed_p90_after_m_s=p90v1,
                )
            )
    return admitted, records
