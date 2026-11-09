"""Plan bounded repairs of isolated object-depth spikes in monocular estimates.

The input medians must come from the same masked object across a fixed frame
timeline. This returns hypotheses; callers must record source identities and
rerun image, depth, motion and physical checks before promoting a reference.
"""

import numpy as np
from scipy.ndimage import median_filter


def plan_masked_depth_spikes(medians_m, areas_px, centers_xy_px, *,
                             window=21, residual_m=.12, max_gap_frames=16,
                             max_anchor_delta_m=.5, min_area_px=500,
                             max_center_speed_px_per_frame=18,
                             max_shift_m=4):
    depth = np.asarray(medians_m, float)
    area = np.asarray(areas_px)
    center = np.asarray(centers_xy_px, float)
    limits = np.asarray([residual_m, max_anchor_delta_m,
                         max_center_speed_px_per_frame, max_shift_m], float)
    if (depth.ndim != 1 or len(depth) < 3 or area.shape != depth.shape
            or center.shape != (len(depth), 2) or window < 3 or window % 2 != 1
            or max_gap_frames < 2 or min_area_px < 1 or
            not np.isfinite(limits).all() or (limits <= 0).any()):
        raise ValueError('Invalid masked-depth repair inputs or limits')
    valid = np.isfinite(depth) & (depth > .2) & np.isfinite(center).all(axis=1)
    if valid.sum() < len(depth) * .9:
        raise ValueError('Insufficient observed object depth for temporal repair')
    filled = depth.copy()
    filled[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), depth[valid])
    baseline = median_filter(filled, size=window, mode='nearest')
    outlier = np.abs(filled - baseline) > residual_m
    plans = []
    for frame in np.flatnonzero(outlier):
        before = np.flatnonzero(~outlier[:frame] & valid[:frame])
        after = np.flatnonzero(~outlier[frame + 1:] & valid[frame + 1:]) + frame + 1
        if not len(before) or not len(after) or not valid[frame]:
            continue
        left, right = int(before[-1]), int(after[0])
        if (right - left > max_gap_frames or
                abs(filled[left] - filled[right]) > max_anchor_delta_m or
                area[frame] < min_area_px or
                np.linalg.norm(center[frame] - center[left]) >
                max_center_speed_px_per_frame * (frame - left) or
                np.linalg.norm(center[frame] - center[right]) >
                max_center_speed_px_per_frame * (right - frame)):
            continue
        target = float(filled[left] + (filled[right] - filled[left]) *
                       (frame - left) / (right - left))
        shift = target - float(filled[frame])
        if abs(shift) > max_shift_m:
            continue
        plans.append(dict(frame=int(frame), original_median_m=float(filled[frame]),
                          repaired_median_m=target, shift_m=shift,
                          left_anchor_frame=left, right_anchor_frame=right))
    return plans
