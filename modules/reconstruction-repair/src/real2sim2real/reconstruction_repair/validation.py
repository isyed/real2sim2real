"""Whole-proposal acceptance. No oracle scores or episode-specific thresholds."""

import numpy as np

from .bundle import geometry_validate
from .contracts import ReconstructionBundle, RepairConfig, RepairProposal
from .geometry import mesh_audit, mesh_surface_equal, reliable_frames
from .scoring import ScoreReport


def geometry_changed_frames(a: dict, b: dict) -> np.ndarray:
    n = len(a["frame_ids"])
    if not np.array_equal(a["object_scale"], b["object_scale"]):
        return np.ones(n, bool)
    if any(not np.array_equal(a[k], b[k]) for k in ("vertices", "faces")) and not mesh_surface_equal(a, b):
        return np.ones(n, bool)
    return (np.max(np.abs(a["object_rotation"] - b["object_rotation"]), axis=(1, 2)) > 1e-9) | (
        np.max(np.abs(a["object_translation"] - b["object_translation"]), axis=1) > 1e-9)


def proposal_validate(bundle: ReconstructionBundle, parent: dict, proposal: RepairProposal,
                      before: ScoreReport, after: ScoreReport, config: RepairConfig) -> dict:
    g, o = proposal.geometry, bundle.observations
    geometry_validate(g)
    changed = geometry_changed_frames(parent, g)
    reasons = []
    benefit = []
    for key in ("frame_ids", "timestamps_s", "intrinsics", "world_to_camera"):
        if not np.array_equal(parent[key], g[key]):
            reasons.append(f"immutable_{key}_changed")
    oldmesh, newmesh = mesh_audit(parent), mesh_audit(g)
    if newmesh["nonmanifold_edges"] > oldmesh["nonmanifold_edges"]:
        reasons.append("nonmanifold_edges_increased")
    if newmesh["used_vertices"] == 0:
        reasons.append("no_mesh_surface")
    if proposal.name == "topology":
        # This proposer only removes exact duplicates and zero-area faces. It
        # must preserve every nondegenerate oriented triangle geometrically.
        if not mesh_surface_equal(parent, g):
            reasons.append("mesh_surface_changed")
        for key in ("object_rotation", "object_translation", "object_scale"):
            if not np.array_equal(parent[key], g[key]):
                reasons.append("mesh_cleanup_changed_pose_or_scale")
        benefit.append("exact_surface_preserving_cleanup")
    else:
        if np.any(changed & ~reliable_frames(o, len(changed))):
            reasons.append("changed_frames_without_reliable_visible_masks")
        if np.any(changed & o.get("locked", np.zeros(len(changed), bool)).astype(bool)):
            reasons.append("locked_frames_changed")
        if np.any(after.per_frame["near_plane_invalid"] & changed):
            reasons.append("candidate_crosses_camera_near_plane")
        valid = np.isfinite(before.per_frame["iou"]) & np.isfinite(after.per_frame["iou"])
        if not valid.any() or np.any(changed & ~valid):
            reasons.append("incomplete_image_review")
        if valid.any():
            delta = after.per_frame["iou"][valid] - before.per_frame["iou"][valid]
            if delta.mean() < -config.max_mean_iou_drop:
                reasons.append("mean_silhouette_regressed")
            if delta.min() < -config.max_frame_iou_drop:
                reasons.append("per_frame_silhouette_regressed")
            if delta.mean() >= config.min_iou_gain:
                benefit.append("silhouette_gain")
        for metric in ("center_acceleration_m_s2", "angular_speed_deg_s"):
            a, b = before.summary[metric], after.summary[metric]
            for stat in ("p95", "max"):
                if a[stat] is not None and b[stat] > max(a[stat] * config.motion_regression_ratio, 1e-6):
                    reasons.append(f"{metric}_{stat}_regressed")
            if a["max"] is not None and a["max"] > 1e-3 and b["max"] <= a["max"] * 0.9:
                benefit.append(f"{metric}_reduced")
        if "hand_points_world" not in o:
            if config.require_contact_evidence:
                reasons.append("contact_evidence_missing")
        else:
            gap_delta = after.per_frame["hand_gap_m"] - before.per_frame["hand_gap_m"]
            # All supplied hand points are guarded; labelled contact gets an
            # additional report, not an exemption for unlabelled frames.
            if gap_delta.max() > config.max_contact_gap_increase_m:
                reasons.append("hand_surface_separation_regressed")
        a, b = before.summary["semantic_facing_agreement"], after.summary["semantic_facing_agreement"]
        if a["mean"] is not None:
            previous = before.per_frame["semantic_agreement"]
            following = after.per_frame["semantic_agreement"]
            if b["mean"] < a["mean"] or np.any((previous == 1) & (following != 1)):
                reasons.append("semantic_orientation_regressed")
            if b["mean"] > a["mean"]:
                benefit.append("annotated_orientation_gain")
        a, b = before.summary["scale_absolute_error_m"], after.summary["scale_absolute_error_m"]
        if a is not None and b < a - 1e-6:
            benefit.append("measured_size_gain")
        if not benefit:
            reasons.append("no_declared_benefit")
    checked = []
    def check(code, why, applicable=True, observed=None, threshold=None):
        checked.append(dict(id=code, status=('failed' if code in reasons else 'passed') if applicable else 'not_checked',
                            explanation=why, observed=observed, threshold=threshold))
    for key in ('frame_ids','timestamps_s','intrinsics','world_to_camera'):
        check('immutable_'+key+'_changed', 'Preserve input '+key)
    check('no_mesh_surface', 'Nonempty finite mesh surface')
    check('nonmanifold_edges_increased', 'No increase in nonmanifold mesh edges')
    if proposal.name == 'topology':
        check('mesh_surface_changed', 'Exact surface preservation')
        check('mesh_cleanup_changed_pose_or_scale', 'Topology cleanup preserves pose and size')
    else:
        check('locked_frames_changed', 'Preserve locked frames')
        check('candidate_crosses_camera_near_plane', 'Keep geometry in front of the camera')
        check('changed_frames_without_reliable_visible_masks', 'Changed frames have reliable image evidence')
        check('incomplete_image_review', 'Complete image review of changed frames')
        check('mean_silhouette_regressed', 'Mean image overlap preservation', valid.any(),
              float(delta.mean()) if valid.any() else None, -config.max_mean_iou_drop)
        check('per_frame_silhouette_regressed', 'Per-frame image overlap preservation', valid.any(),
              float(delta.min()) if valid.any() else None, -config.max_frame_iou_drop)
        for metric in ('center_acceleration_m_s2','angular_speed_deg_s'):
            for stat in ('p95','max'):
                check(f'{metric}_{stat}_regressed', f'Preserve {metric} {stat}', before.summary[metric][stat] is not None,
                      after.summary[metric][stat], config.motion_regression_ratio)
        check('contact_evidence_missing', 'Required hand/contact evidence is present', config.require_contact_evidence)
        check('hand_surface_separation_regressed', 'Hand-to-object separation preservation', 'hand_points_world' in o)
        check('semantic_orientation_regressed', 'Preserve previously correct semantic orientation',
              before.summary['semantic_facing_agreement']['mean'] is not None)
        check('no_declared_benefit', 'At least one declared improvement')
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "benefits": sorted(set(benefit)),
            "checked_gates": checked,
            "changed_frame_ids": parent["frame_ids"][changed].tolist(),
            "limits": "Local evidence gates do not certify depth, penetration, correct hand contact or leaderboard improvement."}
