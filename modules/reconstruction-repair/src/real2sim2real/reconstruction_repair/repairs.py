"""Bounded proposals distilled from source workflow; selection lives in validation.py."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .contracts import ReconstructionBundle, RepairConfig, RepairProposal
from .geometry import (frame_overlap, mesh_center, mesh_silhouette, object_centers,
                       rotation_angle, rotation_between, rotation_interpolate, valid_segments)

from .scoring import motion_statistics

STAGES = ("topology", "scale", "semantic_orientation", "rotation_spikes", "pose_recovery", "temporal_translation")


def _copy(g: dict) -> dict:
    return {key: value.copy() for key, value in g.items()}


def topology_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    vertices, inverse = np.unique(g["vertices"], axis=0, return_inverse=True)
    faces = inverse[g["faces"]]
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    # Only exactly degenerate triangles are removed. No hulls, hole filling,
    # component deletion, approximate welding, recentering or decimation.
    faces = faces[np.any(cross != 0, axis=1)]
    if not len(faces):
        return RepairProposal("topology", candidate, "No nondegenerate surface; manual mesh repair required")
    # Same-direction cyclic duplicates only; preserve opposite-oriented surfaces.
    canonical = np.stack([np.roll(f, -int(np.argmin(f))) for f in faces])
    _, first = np.unique(canonical, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    used = np.unique(faces)
    if len(used) == len(g["vertices"]) and len(faces) == len(g["faces"]):
        return RepairProposal("topology", candidate, "No exact mesh cleanup needed; preserve original indexing")
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    candidate["vertices"], candidate["faces"] = vertices[used], remap[faces].astype(g["faces"].dtype)
    return RepairProposal("topology", candidate, "Exact weld and duplicate/zero-area removal; surface coordinates preserved")


def scale_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    evidence = bundle.manifest.get("evidence", {}).get("scale")
    if not evidence:
        return RepairProposal("scale", candidate, "No measured/calibrated size evidence; preserve scale")
    axis = evidence["axis"]
    extent = np.ptp(g["vertices"][np.unique(g["faces"])], axis=0)[axis] * g["object_scale"]
    ratio = evidence["length_m"] / max(float(extent), 1e-12)
    if abs(ratio - 1) < 1e-10:
        return RepairProposal("scale", candidate, "Measured size already matches")
    if not 1 / config.max_scale_ratio <= ratio <= config.max_scale_ratio:
        return RepairProposal("scale", candidate, "Size evidence exceeds the declared correction bound", {"proposed_ratio": ratio})
    centers = object_centers(g)
    candidate["object_scale"] = np.asarray(g["object_scale"] * ratio)
    candidate["object_translation"] = centers - np.einsum("tij,j->ti", g["object_rotation"], mesh_center(candidate))
    return RepairProposal("scale", candidate, "Evidence-based scale; preserve world-space mesh center", {"ratio": ratio, "evidence": evidence})


def semantic_orientation_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    evidence = bundle.manifest.get("evidence", {})
    axis = evidence.get("semantic_axis_local")
    labels = bundle.observations.get("axis_facing")
    if axis is None or labels is None or np.count_nonzero(labels) < 3:
        return RepairProposal("semantic_orientation", candidate, "Need a local semantic axis and at least three facing labels")
    axis = np.asarray(axis, dtype=float)
    flip = rotation_between(axis, -axis)
    candidate["object_rotation"] = g["object_rotation"] @ flip
    # A whole-sequence branch avoids inventing a jump at isolated annotations.
    return RepairProposal("semantic_orientation", candidate,
                          "Assisted whole-sequence opening/face flip; translation, mesh and human fixed",
                          {"mode": "assisted", "labelled_frames": int(np.count_nonzero(labels))})


def rotation_spikes_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    detected = []
    for segment in valid_segments(g, bundle.observations, config.max_gap_s):
        for i in segment[1:-1]:
            left, right = g["object_rotation"][i - 1], g["object_rotation"][i + 1]
            if (np.rad2deg(rotation_angle(left, right)) <= config.neighbor_angle_deg and
                min(np.rad2deg(rotation_angle(left, g["object_rotation"][i])),
                    np.rad2deg(rotation_angle(right, g["object_rotation"][i]))) >= config.spike_angle_deg):
                dt = g["timestamps_s"][i + 1] - g["timestamps_s"][i - 1]
                ratio = (g["timestamps_s"][i] - g["timestamps_s"][i - 1]) / dt
                candidate["object_rotation"][i] = rotation_interpolate(left, right, float(ratio))
                center = object_centers(g)[i]
                candidate["object_translation"][i] = center - candidate["object_rotation"][i] @ mesh_center(g)
                detected.append(int(g["frame_ids"][i]))
    return RepairProposal("rotation_spikes", candidate, "Time-weighted geodesic replacement of isolated reversing excursions",
                          {"detected_frame_ids": detected, "pivot": "world mesh center"})


def _pose_fit(bundle: ReconstructionBundle, g: dict, i: int, config: RepairConfig, neighbors: list[int]) -> tuple:
    o = bundle.observations
    w = g["world_to_camera"][i]
    rotation = w[:3, :3] @ g["object_rotation"][i]
    origin = w[:3, :3] @ g["object_translation"][i] + w[:3, 3]
    bound = np.deg2rad(config.pose_rotation_bound_deg)
    candidate = _copy(g)

    def assign(x):
        candidate["object_rotation"][i] = w[:3, :3].T @ Rotation.from_rotvec(x[:3]).as_matrix() @ rotation
        candidate["object_translation"][i] = w[:3, :3].T @ (origin + [x[3], x[4], 0] - w[:3, 3])

    def objective(x):
        if np.linalg.norm(x[:3]) > bound or np.linalg.norm(x[3:]) > config.pose_xy_bound_m:
            return 2.0 + float(np.linalg.norm(x))
        assign(x)
        return 1 - frame_overlap(candidate, o, i, config.fit_image_size, crop_to_object=config.fit_crop_to_object) + 1e-5 * float(x @ x)

    seeds = [np.zeros(5)]
    target = o["object_mask"][i].astype(bool)
    if "occluder_mask" in o:
        target &= ~o["occluder_mask"][i].astype(bool)
    render = mesh_silhouette(g, i, *target.shape)
    if target.any() and render.any():
        delta_px = np.argwhere(target).mean(axis=0)[::-1] - np.argwhere(render).mean(axis=0)[::-1]
        camera_center = w[:3, :3] @ object_centers(g)[i] + w[:3, 3]
        xy = (np.linalg.inv(g["intrinsics"][i]) @ np.r_[delta_px, 0])[:2] * camera_center[2]
        xy *= min(1, config.pose_xy_bound_m * 0.95 / max(np.linalg.norm(xy), 1e-12))
        seeds.append(np.r_[np.zeros(3), xy])
    for j in neighbors:
        delta = Rotation.from_matrix(w[:3, :3] @ g["object_rotation"][j] @ rotation.T).as_rotvec()
        if np.linalg.norm(delta) <= bound:
            seeds.append(np.r_[delta, seeds[-1][3:]])
    # Optional ring-plane evidence supplies hypotheses, never a hemisphere sign
    # certification. Ellipse fit is approximate under perspective/occlusion.
    profile = bundle.manifest.get("object_profile", {})
    if profile.get("shape") == "ring" and "plane_normal_local" in profile and target.sum() > 20:
        xy = np.argwhere(target)[:, ::-1]
        normalized = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(g["intrinsics"][i]).T
        values, vectors = np.linalg.eigh(np.cov(normalized[:, :2].T))
        cosine = np.sqrt(max(values[0], 0) / max(values[1], 1e-12))
        normal_xy = vectors[:, 0] * np.sqrt(max(0, 1 - cosine**2))
        base_axis = rotation @ np.asarray(profile["plane_normal_local"], dtype=float)
        for sign in (-1, 1):
            normal = np.r_[sign * normal_xy, cosine]
            if normal @ base_axis < 0:
                normal = -normal
            delta = Rotation.from_matrix(rotation_between(base_axis, normal)).as_rotvec()
            if np.linalg.norm(delta) <= bound:
                seeds.append(np.r_[delta, np.zeros(2)])
    start = min(seeds, key=objective)
    bounds = [(-bound, bound)] * 3 + [(-config.pose_xy_bound_m, config.pose_xy_bound_m)] * 2
    result = minimize(objective, start, method="Powell", bounds=bounds,
                      options={"maxfev": config.fit_max_evaluations, "xtol": 1e-4, "ftol": 1e-5})
    best = min([*seeds, result.x], key=objective)
    assign(best)
    full_iou = frame_overlap(candidate, o, i)
    translation_only = best.copy()
    translation_only[:3] = 0
    assign(translation_only)
    if frame_overlap(candidate, o, i) >= full_iou:
        best = translation_only  # do not invent rotation when translation explains the same pixels
    assign(best)
    if frame_overlap(candidate, o, i) < frame_overlap(g, o, i) + config.min_iou_gain:
        return np.eye(3), np.zeros(2)
    return candidate["object_rotation"][i] @ g["object_rotation"][i].T, best[3:]


def limit_pose_motion(parent: dict, proposal: dict, config: RepairConfig) -> tuple[dict, float]:
    """Backtrack the entire correction field before expensive image/contact review.

    This is not acceptance: complete evidence and contact guards still run.
    A common scalar keeps camera-XY displacement bounds and fixed frames intact;
    SO(3) interpolation keeps proper rotations. No clean reference is involved.
    """
    baseline = motion_statistics(parent, config.max_gap_s)
    delta_rotation = Rotation.from_matrix(proposal["object_rotation"] @ parent["object_rotation"].swapaxes(-1, -2)).as_rotvec()
    unchanged_rotation = np.max(np.abs(proposal["object_rotation"]-parent["object_rotation"]), axis=(1, 2)) < 1e-12
    delta_translation = proposal["object_translation"]-parent["object_translation"]
    for step in range(config.pose_backtracking_steps):
        fraction = 0.5 ** step
        candidate = dict(proposal)
        candidate["object_rotation"] = Rotation.from_rotvec(delta_rotation*fraction).as_matrix() @ parent["object_rotation"]
        candidate["object_rotation"][unchanged_rotation] = parent["object_rotation"][unchanged_rotation]
        candidate["object_translation"] = parent["object_translation"]+fraction*delta_translation
        measured = motion_statistics(candidate, config.max_gap_s)
        okay = all(baseline[key][stat] is None or measured[key][stat] <= max(baseline[key][stat]*config.motion_regression_ratio, 1e-6)
                   for key in ("center_acceleration_m_s2", "angular_speed_deg_s") for stat in ("max", "p95"))
        if okay:
            return candidate, fraction
    return _copy(parent), 0.0


def pose_recovery_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    anchor_ids = []
    segments = valid_segments(g, bundle.observations, config.max_gap_s)
    overlap_cache = {}
    def overlap(i):
        if i not in overlap_cache:
            overlap_cache[i] = frame_overlap(g, bundle.observations, int(i))
        return overlap_cache[i]
    eligible = []
    for segment_index, segment in enumerate(segments):
        if len(segment) < 3:
            continue
        last_time = -np.inf
        for i in segment[1:-1]:
            if g["timestamps_s"][i] - last_time >= config.anchor_spacing_s and overlap(i) < config.pose_trigger_iou:
                eligible.append((segment_index, int(i)))
                last_time = g["timestamps_s"][i]
    if len(eligible) > config.max_fit_anchors:
        eligible = [eligible[i] for i in np.linspace(0, len(eligible)-1, config.max_fit_anchors).astype(int)]
    for segment_index, segment in enumerate(segments):
        sampled = [i for group, i in eligible if group == segment_index]
        if not sampled:
            continue
        # Reliable frames already agreeing with the mask are fixed anchors too.
        # Otherwise interpolation would spread a bad-window correction into
        # neighboring frames that were already correct.
        good = [int(i) for i in segment if overlap(i) >= config.pose_trigger_iou]
        fixed = set([int(segment[0]), int(segment[-1]), *good])
        knots = sorted(fixed | set(sampled))
        fitted = {}
        for i in sampled:
            neighbors = [int(segment[0]), int(segment[-1])]
            fitted[i] = _pose_fit(bundle, g, i, config, neighbors)
        changes = [fitted.get(i, (np.eye(3), np.zeros(2))) for i in knots]
        for left in range(len(knots) - 1):
            a, b = knots[left:left + 2]
            for i in range(a + 1, b + 1):
                alpha = (g["timestamps_s"][i] - g["timestamps_s"][a]) / (g["timestamps_s"][b] - g["timestamps_s"][a])
                delta = rotation_interpolate(changes[left][0], changes[left + 1][0], float(alpha))
                xy = (1 - alpha) * changes[left][1] + alpha * changes[left + 1][1]
                if np.max(np.abs(delta - np.eye(3))) < 1e-12 and np.linalg.norm(xy) < 1e-12:
                    continue
                candidate["object_rotation"][i] = delta @ g["object_rotation"][i]
                candidate["object_translation"][i] = g["object_translation"][i] + g["world_to_camera"][i, :3, :3].T @ np.r_[xy, 0]
        # Preserve both segment endpoints bit-for-bit.
        for i in fixed:
            candidate["object_rotation"][i] = g["object_rotation"][i]
            candidate["object_translation"][i] = g["object_translation"][i]
        anchor_ids.extend(int(g["frame_ids"][i]) for i in sampled)
    fraction = 1.0
    if config.pose_motion_backtracking:
        candidate, fraction = limit_pose_motion(g, candidate, config)
    return RepairProposal("pose_recovery", candidate, "Bounded mask fit with neighbor/plane hypotheses; camera-origin depth fixed",
                          {"fit_anchor_frame_ids": anchor_ids,
                           "anchor_allocation": "uniform coverage of eligible anchors across all valid segments",
                           "fit_crop_to_object": config.fit_crop_to_object,
                           "motion_backtracking_fraction": fraction,
                           "review": "all supplied frames, including non-fit frames; unchanged acceptance gates"})


def temporal_translation_propose(bundle: ReconstructionBundle, g: dict, config: RepairConfig) -> RepairProposal:
    candidate = _copy(g)
    centers = object_centers(g)
    ts = g["timestamps_s"]
    for segment in valid_segments(g, bundle.observations, config.max_gap_s):
        interior = segment[1:-1]
        if len(interior) < 2:
            continue
        changes = []
        for i in interior:
            window = segment[np.abs(ts[segment] - ts[i]) <= config.smoothing_window_s]
            # Fit a local line in time: constant velocity survives irregular sampling.
            t = ts[window] - ts[i]
            weights = np.exp(-0.5 * (t / (config.smoothing_window_s / 2)) ** 2)
            design = np.c_[np.ones(len(t)), t] * weights[:, None] ** 0.5
            estimate = np.linalg.lstsq(design, centers[window] * weights[:, None] ** 0.5, rcond=None)[0][0]
            changes.append(estimate - centers[i])
        changes = np.asarray(changes)
        if config.temporal_preserve_camera_depth:
            z_axis = g["world_to_camera"][interior, 2, :3]
            projection = np.eye(3) - np.einsum("ni,nj->nij", z_axis, z_axis)
            changes = np.einsum("nij,nj->ni", projection, changes)
            # Project simultaneously onto camera XY planes and sum(delta)=0,
            # including moving cameras. A common scalar clamp preserves both.
            lagrange = np.linalg.pinv(projection.sum(axis=0)) @ changes.sum(axis=0)
            changes -= np.einsum("nij,j->ni", projection, lagrange)
        else:
            changes -= changes.mean(axis=0)
        if np.max(np.abs(changes)) < 1e-12:
            continue
        changes *= min(1, config.translation_bound_m / max(np.linalg.norm(changes, axis=1).max(), 1e-12))
        candidate["object_translation"][interior] += changes
    return RepairProposal("temporal_translation", candidate,
                          "Time-aware bounded zero-mean translation filter; segment endpoints/occlusions/contact transitions preserved",
                          {"camera_depth_preserved": config.temporal_preserve_camera_depth})


PROPOSERS = {name: globals()[f"{name}_propose"] for name in STAGES}
