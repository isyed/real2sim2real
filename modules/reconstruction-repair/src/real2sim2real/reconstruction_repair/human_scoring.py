"""Human diagnostics and whole-sequence acceptance, without hidden references."""

import numpy as np

from .geometry import mesh_silhouette, object_world_vertices, points_mesh_distance
from .human_proposals import human_controls, human_segments, wrap
from .scoring import _stats


def human_scores(bundle, geometry, decoded, decoder, config):
    g, o = geometry, bundle.observations
    n = len(g["frame_ids"])
    camera = np.einsum("tij,tkj->tki", g["world_to_camera"][:, :3, :3], decoded.joints_world) + g["world_to_camera"][:, None, :3, 3]
    projection = np.einsum("tij,tkj->tki", g["intrinsics"], camera)
    xy = projection[:, :, :2] / np.maximum(projection[:, :, 2:], 1e-8)
    confidence = o["human_keypoints_confidence"] >= config.human_confidence_min
    confidence[:, :5] = False
    valid = o["human_reliable"].astype(bool) & (confidence.sum(axis=1) >= config.human_min_joints)
    iou, error = np.full(n, np.nan), np.full(n, np.nan)
    near_plane = np.zeros(n, bool)
    temporary = {**g, "object_rotation": np.tile(np.eye(3), (n, 1, 1)), "object_translation": np.zeros((n, 3)),
                 "object_scale": np.array(1.), "faces": decoder.faces}
    gaps = np.zeros((n, 2))
    for i in range(n):
        camera_vertices = decoded.vertices_world[i] @ g["world_to_camera"][i, :3, :3].T + g["world_to_camera"][i, :3, 3]
        near_plane[i] = np.any(camera_vertices[:, 2] <= 1e-5)
        if valid[i] and not near_plane[i]:
            target = o["human_mask"][i].astype(bool)
            occ = o.get("human_occluder_mask")
            exclude = np.zeros_like(target) if occ is None else occ[i].astype(bool)
            target &= ~exclude
            if target.any():
                temporary["vertices"] = decoded.vertices_world[i]
                render = mesh_silhouette(temporary, i, *target.shape) & ~exclude
                iou[i] = np.count_nonzero(target & render) / max(1, np.count_nonzero(target | render))
                error[i] = np.linalg.norm(xy[i]-o["human_keypoints_xy"][i], axis=1)[confidence[i]].mean()
        for side, indices in enumerate(decoder.hand_vertices):
            chosen = indices[np.linspace(0, len(indices)-1, min(16, len(indices))).astype(int)]
            gaps[i, side] = points_mesh_distance(decoded.vertices_world[i, chosen], object_world_vertices(g, i), g["faces"]).min()
    dt = np.diff(g["timestamps_s"])
    velocity = np.diff(decoded.joints_world, axis=0) / dt[:, None, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1]+dt[1:])*.5)[:, None, None]
    pair = dt <= config.max_gap_s
    summary = {"silhouette_iou": _stats(iou), "joint_error_px": _stats(error),
               "camera_joint_depth_m": _stats(camera[:, :, 2]),
               "joint_speed_m_s": _stats(np.linalg.norm(velocity[pair], axis=-1)),
               "joint_acceleration_m_s2": _stats(np.linalg.norm(acceleration[pair[:-1] & pair[1:]], axis=-1)),
               "sampled_hand_gap_m": _stats(gaps), "image_reviewed_frames": int(np.isfinite(iou).sum())}
    fit = o.get("human_fit_frames", np.zeros(n, bool)).astype(bool)
    summary["fit_joint_error_px"], summary["holdout_joint_error_px"] = _stats(error[fit]), _stats(error[~fit])
    return {"summary": summary, "iou": iou, "error": error, "gaps": gaps, "near_plane": near_plane,
            "camera_joints": camera}


def human_comparison(before, after):
    return {"before": before["summary"], "after": after["summary"],
            "absolute_camera_joint_depth_change_m": _stats(np.abs(after["camera_joints"][:, :, 2]-before["camera_joints"][:, :, 2])),
            "after_minus_before": {key: {stat: (after["summary"][key][stat] - value if value is not None and after["summary"][key][stat] is not None else None)
                                                  for stat, value in item.items()}
                                   for key, item in before["summary"].items() if isinstance(item, dict)},
            "limits": "Diagnostics, not official CD-H/ACC-H/PEN. Sampled unsigned hand gaps do not certify contact or penetration."}


def human_validate(bundle, original, candidate, original_decoded, candidate_decoded, before, after, decoder, config, fitting=False):
    g, o = bundle.geometry, bundle.observations
    reasons, benefits = [], []
    allowed = np.r_[0:3, human_controls(decoder)]
    frozen = np.setdiff1d(np.arange(136), allowed)
    if not np.array_equal(original.pose[:, frozen], candidate.pose[:, frozen]):
        reasons.append("protected_root_orientation_fingers_or_lengths_changed")
    if any(not np.array_equal(a, b) for a, b in [(original.shape, candidate.shape), (original.scales, candidate.scales), (original.expression, candidate.expression)]):
        reasons.append("human_identity_changed")
    changed = np.any(candidate.pose != original.pose, axis=1)
    editable = np.zeros(len(changed), bool)
    for segment in human_segments(bundle, config):
        editable[segment[1:-1]] = True
    if np.any(changed & ~editable):
        reasons.append("anchor_unknown_or_transition_frame_changed")
    if np.any(after["near_plane"] & changed):
        reasons.append("human_crosses_near_plane")
    valid = np.isfinite(before["iou"]) & np.isfinite(after["iou"]) & np.isfinite(before["error"]) & np.isfinite(after["error"])
    if np.any(changed & ~valid) or not valid.any():
        reasons.append("incomplete_human_image_evidence")
    if valid.any():
        delta = after["iou"][valid]-before["iou"][valid]
        if delta.mean() < -config.max_mean_iou_drop or delta.min() < -config.max_frame_iou_drop:
            reasons.append("human_silhouette_regressed")
        error = after["error"][valid]-before["error"][valid]
        if error.mean() > 1e-4 or error.max() > config.human_max_joint_regression_px:
            reasons.append("human_keypoints_regressed")
        evaluate = valid & ~o.get("human_fit_frames", np.zeros(len(valid), bool)).astype(bool) if fitting else valid
        if not evaluate.any():
            reasons.append("no_human_fit_holdout")
        else:
            gain = before["error"][evaluate].mean()-after["error"][evaluate].mean()
            if gain >= config.human_min_joint_gain_px and gain >= before["error"][evaluate].mean()*config.human_min_joint_gain_fraction:
                benefits.append("heldout_joint_gain" if fitting else "joint_gain")
            if not fitting and delta.mean() >= config.min_iou_gain:
                benefits.append("human_silhouette_gain")
    if np.linalg.norm(candidate.pose[:, :3]-original.pose[:, :3], axis=1).max()*.1 > config.human_root_bound_m + 1e-6:
        reasons.append("human_root_bound")
    if np.abs(wrap(candidate.pose[:, allowed[3:]]-original.pose[:, allowed[3:]])).max() > np.deg2rad(config.human_angle_bound_deg) + 1e-6:
        reasons.append("human_angle_bound")
    joint_shift = np.linalg.norm(candidate_decoded.joints_world-original_decoded.joints_world, axis=-1).max()
    depth_shift = np.abs(after["camera_joints"][:, :, 2]-before["camera_joints"][:, :, 2]).max()
    if joint_shift > config.human_joint_bound_m:
        reasons.append("decoded_joint_displacement_bound")
    if depth_shift > config.human_depth_bound_m:
        reasons.append("human_depth_drift_bound")
    hand_change = []
    for side, indices in enumerate(decoder.hand_vertices):
        displacement = np.linalg.norm(candidate_decoded.vertices_world[:, indices]-original_decoded.vertices_world[:, indices], axis=-1).max(axis=1)
        limit = np.where(o["human_contact_active"][:, side], min(config.human_hand_bound_m, config.max_contact_gap_increase_m), config.human_hand_bound_m)
        if np.any(displacement > limit + 1e-6):
            reasons.append(f"hand_{side}_displacement_bound")
        hand_change.append(float(displacement.max()))
    if (after["gaps"]-before["gaps"]).max() > config.max_contact_gap_increase_m:
        reasons.append("human_hand_separation_regressed")
    for metric in ("joint_speed_m_s", "joint_acceleration_m_s2"):
        for stat in ("p95", "max"):
            a, b = before["summary"][metric][stat], after["summary"][metric][stat]
            # Decoder float32 noise has a nonzero numerical floor; this is
            # recorded here, not tuned per episode or after a fit.
            floor = .001 if metric == "joint_speed_m_s" else .03
            if a is not None and b > max(a*config.motion_regression_ratio, floor):
                reasons.append(metric+"_"+stat+"_regressed")
        a, b = before["summary"][metric]["max"], after["summary"][metric]["max"]
        if not fitting and a is not None and a > .03 and b < a*.9:
            benefits.append(metric+"_reduced")
    if not benefits:
        reasons.append("no_declared_human_benefit")
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "benefits": sorted(set(benefits)),
            "changed_frame_ids": g["frame_ids"][changed].tolist(), "max_joint_displacement_m": float(joint_shift),
            "max_depth_shift_m": float(depth_shift), "max_hand_displacement_m": hand_change}
