"""Bounded native scalar-control proposals; never smooth identity or fingers."""

import numpy as np

from .human_contract import REPAIR_CONTROLS


def human_controls(decoder) -> np.ndarray:
    return np.asarray([decoder.parameter_names.index(name) for name in REPAIR_CONTROLS])


def human_segments(bundle, config) -> list[np.ndarray]:
    o, g = bundle.observations, bundle.geometry
    n = len(g["frame_ids"])
    valid = o["human_reliable"].astype(bool) & o["human_mask"].any(axis=(1, 2))
    valid &= (o["human_keypoints_confidence"][:, 5:] >= config.human_confidence_min).sum(axis=1) >= config.human_min_joints
    valid &= ~o.get("human_locked", np.zeros(n, bool)).astype(bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    split = (np.diff(ids) != 1) | (np.diff(g["timestamps_s"][ids]) > config.max_gap_s)
    split |= np.any(o["human_contact_active"][ids[1:]] != o["human_contact_active"][ids[:-1]], axis=1)
    return list(np.split(ids, np.flatnonzero(split) + 1))


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def human_bounded_pose(original, candidate, controls, config):
    out = original.copy()
    root = candidate[:, :3] - original[:, :3]
    length = np.linalg.norm(root, axis=1, keepdims=True)
    root *= np.minimum(1, config.human_root_bound_m / .1 / np.maximum(length, 1e-12))
    out[:, :3] += root
    delta = np.clip(wrap(candidate[:, controls] - original[:, controls]),
                    -np.deg2rad(config.human_angle_bound_deg), np.deg2rad(config.human_angle_bound_deg))
    out[:, controls] += delta
    out[np.max(np.abs(out - original), axis=1) < 1e-7] = original[np.max(np.abs(out - original), axis=1) < 1e-7]
    return out


def human_motion_proposal(bundle, parameters, decoder, config, kind):
    original = parameters.pose
    candidate = original.copy()
    controls = human_controls(decoder)
    ts = bundle.geometry["timestamps_s"]
    for segment in human_segments(bundle, config):
        for i in segment[1:-1]:
            if kind == "outliers":
                a, b = i - 1, i + 1
                alpha = (ts[i] - ts[a]) / (ts[b] - ts[a])
                root_agrees = np.linalg.norm(original[a, :3] - original[b, :3]) * .1 < config.human_root_spike_m / 3
                root_spike = min(np.linalg.norm(original[i, :3] - original[a, :3]),
                                 np.linalg.norm(original[i, :3] - original[b, :3])) * .1 > config.human_root_spike_m
                if root_agrees and root_spike:
                    candidate[i, :3] = (1-alpha)*original[a, :3] + alpha*original[b, :3]
                agrees = np.abs(wrap(original[a, controls] - original[b, controls])) < np.deg2rad(config.human_angle_spike_deg / 3)
                spike = np.minimum(np.abs(wrap(original[i, controls]-original[a, controls])),
                                   np.abs(wrap(original[i, controls]-original[b, controls]))) > np.deg2rad(config.human_angle_spike_deg)
                selected = controls[agrees & spike]
                candidate[i, selected] = original[a, selected] + alpha*wrap(original[b, selected]-original[a, selected])
            else:
                window = segment[np.abs(ts[segment]-ts[i]) <= config.smoothing_window_s]
                time = ts[window]-ts[i]
                weights = np.exp(-.5*(time/(config.smoothing_window_s/2))**2)
                design = np.c_[np.ones(len(time)), time] * np.sqrt(weights[:, None])
                root = np.linalg.lstsq(design, original[window, :3]*np.sqrt(weights[:, None]), rcond=None)[0][0]
                local_angles = original[i, controls] + wrap(original[window[:, None], controls]-original[i, controls])
                angles = np.linalg.lstsq(design, local_angles*np.sqrt(weights[:, None]), rcond=None)[0][0]
                candidate[i, :3], candidate[i, controls] = root, angles
    return human_bounded_pose(original, candidate, controls, config)
