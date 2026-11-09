"""Optional human stages, parity checks, atomic recovery and immutable fallback."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .bundle import arrays_atomic, arrays_read, asset_path, file_sha256, json_atomic, _rigid_rotations
from .human_contract import human_parameters, human_select, human_with_pose
from .human_proposals import human_motion_proposal
from .human_scoring import human_scores, human_validate, human_comparison


@dataclass
class HumanRepairResult:
    arrays: dict
    changed: bool
    report: dict


def human_preflight(bundle, decoder, config):
    if bundle.manifest.get("repair", {}).get("human_applied"):
        raise ValueError("Human repairs were already applied; use the original initializer for a new controlled branch")
    if "human" not in bundle.manifest["assets"]:
        raise ValueError("Human repair requires a native human asset")
    metadata = bundle.manifest["human"]
    if metadata.get("encoding") != "mhr-native-136-v1":
        raise ValueError("Opt-in human repair requires explicit mhr-native-136-v1 encoding")
    for key in ("model_sha256", "buffers_sha256"):
        if metadata.get("decoder_assets", {}).get(key) != decoder.identity[key]:
            raise ValueError("Bundle and runtime native-MHR decoder identities differ")
    transform = np.asarray(metadata.get("native_to_world"), dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Human repair requires an explicit proper native_to_world transform")
    _rigid_rotations(transform[:3, :3], "native_to_world")
    o = bundle.observations
    required = {"human_mask", "human_reliable", "human_keypoints_xy", "human_keypoints_confidence", "human_contact_active"}
    if required - set(o):
        raise ValueError(f"Human repair evidence missing: {sorted(required-set(o))}")
    evidence = bundle.manifest.get("evidence", {})
    if not evidence.get("human_observation_source") or not evidence.get("human_contact_source"):
        raise ValueError("Human image and contact-interval provenance are required")
    raw = arrays_read(asset_path(bundle.root, bundle.manifest["assets"]["human"]))
    allowed = {"pose", "shape", "scales", "expression", "parity_frame_ids", "parity_vertices_world", "vertices_world", "joints_world"}
    if set(raw) - allowed:
        raise ValueError("Unrecognized human payload fields need explicit preservation/refresh semantics")
    parameters = human_parameters(raw, len(bundle.geometry["frame_ids"]))
    if not {"parity_frame_ids", "parity_vertices_world"} <= set(raw) or not metadata.get("parity_source"):
        raise ValueError("Human repair requires independently exported native decoder parity geometry and provenance")
    ids = raw["parity_frame_ids"]
    frame_ids = bundle.geometry["frame_ids"]
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or ids[0] != frame_ids[0] or np.any(np.diff(ids) <= 0) or not np.isin(ids, frame_ids).all():
        raise ValueError("Parity frame IDs must be a sorted input subset including the first frame")
    indices = np.searchsorted(frame_ids, ids)
    decoded = decoder.decode(parameters, transform)
    if (decoded.vertices_world.ndim != 3 or decoded.vertices_world.shape[0] != len(frame_ids)
            or decoded.vertices_world.shape[2] != 3 or decoded.joints_world.shape != (len(frame_ids), 17, 3)
            or not np.isfinite(decoded.vertices_world).all() or not np.isfinite(decoded.joints_world).all()):
        raise ValueError("Native decoder returned invalid geometry")
    target = raw["parity_vertices_world"]
    if target.shape != decoded.vertices_world[indices].shape:
        raise ValueError("Native decoder parity mesh shape differs")
    parity = float(np.max(np.abs(target-decoded.vertices_world[indices])))
    if parity > config.human_parity_tolerance_m:
        raise ValueError(f"Native decoder geometry parity failed: {parity} m")
    # Verify the actual model's root-translation units, independently of a label.
    first = human_select(parameters, np.array([0]))
    for axis in range(3):
        pose = first.pose.copy(); pose[:, axis] += .01
        moved = decoder.decode(human_with_pose(first, pose), transform)
        expected = transform[:3, axis]*.001
        if np.max(np.abs(moved.vertices_world-decoded.vertices_world[:1]-expected)) > max(config.human_parity_tolerance_m, 2e-6):
            raise ValueError("Native decoder root translation convention failed")
    if config.human_fit:
        fit = o.get("human_fit_frames")
        valid = o["human_reliable"].astype(bool) & o["human_mask"].any(axis=(1, 2))
        valid &= (o["human_keypoints_confidence"][:, 5:] >= config.human_confidence_min).sum(axis=1) >= config.human_min_joints
        if fit is None or not 2 <= int(fit.sum()) <= 64 or np.any(fit.astype(bool) & ~valid) or (valid & ~fit.astype(bool)).sum() < 2:
            raise ValueError("Human fitting requires 2–64 reliable fit frames and at least two reliable held-out frames")
    return raw, parameters, decoded, transform, {"max_vertex_error_m": parity, "parity_frame_ids": ids.tolist(), "root_units_verified": True}


def human_stage_run(bundle, selected_geometry, output: Path, config, decoder, run_id: str) -> HumanRepairResult:
    output.mkdir(parents=True, exist_ok=True)
    raw, original, initial_decoded, transform, parity = human_preflight(bundle, decoder, config)
    current, current_decoded = original, initial_decoded
    # Human proposals use the selected object. Final cumulative checks compare
    # BOTH human and object against the original bundle's interaction geometry.
    before_original = human_scores(bundle, bundle.geometry, initial_decoded, decoder, config)
    current_scores = human_scores(bundle, selected_geometry, current_decoded, decoder, config)
    records = []
    parent_sha = bundle.manifest["assets"]["human"]["sha256"]
    for stage, enabled in (("outliers", config.human_outliers), ("temporal", config.human_temporal), ("fit", config.human_fit)):
        checkpoint, receipt = output/f"{stage}.npz", output/f"{stage}.json"
        json_atomic(output/"status.json", {"status": "running", "stage": stage, "run_id": run_id})
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["run_id"] != run_id or record["parent_sha256"] != parent_sha or record["selected_sha256"] != file_sha256(checkpoint):
                raise ValueError("Human stage recovery identity/checksum mismatch")
            current = human_with_pose(original, arrays_read(checkpoint)["pose"])
            current_decoded = decoder.decode(current, transform)
            current_scores = human_scores(bundle, selected_geometry, current_decoded, decoder, config)
        else:
            record = {"stage": "human_"+stage, "run_id": run_id, "parent_sha256": parent_sha, "reason": "Disabled by configuration", "outcome": "disabled"}
            if enabled:
                details = {}
                if stage == "fit":
                    from .human_fit import human_fit_proposal
                    pose, details = human_fit_proposal(bundle, current, current_decoded, decoder, transform, config, output, run_id+parent_sha)
                else:
                    pose = human_motion_proposal(bundle, current, decoder, config, stage)
                record.update(reason="Bounded native-MHR proposal; original shape/fingers/root orientation retained", details=details)
                if np.array_equal(pose, current.pose):
                    record["outcome"] = "unchanged"
                else:
                    proposed = human_with_pose(current, pose)
                    decoded = decoder.decode(proposed, transform)
                    after = human_scores(bundle, selected_geometry, decoded, decoder, config)
                    validation = human_validate(bundle, current, proposed, current_decoded, decoded, current_scores, after, decoder, config, stage == "fit")
                    record.update(validation=validation, scores=human_comparison(current_scores, after), outcome="accepted" if validation["accepted"] else "rejected")
                    if validation["accepted"]:
                        current, current_decoded, current_scores = proposed, decoded, after
            arrays_atomic(checkpoint, {"pose": current.pose})
            record["selected_sha256"] = file_sha256(checkpoint)
            json_atomic(receipt, record)
        parent_sha = record["selected_sha256"]
        records.append(record)
    changed = not np.array_equal(current.pose, original.pose)
    aggregate = None
    if changed:
        aggregate = human_validate(bundle, original, current, initial_decoded, current_decoded,
                                   before_original, current_scores, decoder, config,
                                   any(r["stage"] == "human_fit" and r["outcome"] == "accepted" for r in records))
        if not aggregate["accepted"]:
            current, current_decoded = original, initial_decoded
            current_scores = human_scores(bundle, selected_geometry, current_decoded, decoder, config)
            changed = False
            records = [{**r, "outcome": "superseded_by_final_fallback"} if r["outcome"] == "accepted" else r for r in records]
    # Object changes must also pass decoded-human interaction checks when the
    # human proposal is unchanged/rejected. Otherwise both return to baseline.
    pair_contact_safe = bool((current_scores["gaps"]-before_original["gaps"]).max() <= config.max_contact_gap_increase_m)
    if not pair_contact_safe:
        current, current_decoded, current_scores, changed = original, initial_decoded, before_original, False
        records = [{**r, "outcome": "superseded_by_pair_fallback"} if r["outcome"] == "accepted" else r for r in records]
    selected = {**raw, "pose": current.pose}
    if changed:
        indices = np.searchsorted(bundle.geometry["frame_ids"], raw["parity_frame_ids"])
        selected["parity_vertices_world"] = current_decoded.vertices_world[indices]
        if "vertices_world" in selected: selected["vertices_world"] = current_decoded.vertices_world
        selected["joints_world"] = current_decoded.joints_world
    report = {"schema": "r2s2r_engines.native-human-repair/1", "decoder": decoder.identity, "parity": parity,
              "changed": changed, "stages": records, "aggregate_validation": aggregate,
              "requires_object_fallback": not pair_contact_safe,
              "pair_contact_reason": "Selected pair exceeded original decoded-human hand separation" if not pair_contact_safe else "Within declared unsigned separation limit",
              "scores": human_comparison(before_original, current_scores),
              "preserved": ["shape", "scales", "expression", "fingers", "root_orientation", "flexible_lengths", "segment_anchors"],
              "limitations": ["No official CD-H/ACC-H/PEN claim", "Contact preservation is not contact correctness",
                              "Human fits do not certify calibrated metric depth", "No clean reference is a fitting input"]}
    json_atomic(output/"report.json", report)
    with (output/"per_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_id", "iou_before", "iou_after", "joint_error_before_px", "joint_error_after_px", "left_gap_before_m", "left_gap_after_m", "right_gap_before_m", "right_gap_after_m"])
        for i, frame in enumerate(bundle.geometry["frame_ids"]):
            values = [before_original["iou"][i], current_scores["iou"][i], before_original["error"][i], current_scores["error"][i],
                      before_original["gaps"][i, 0], current_scores["gaps"][i, 0], before_original["gaps"][i, 1], current_scores["gaps"][i, 1]]
            writer.writerow([int(frame), *[float(x) if np.isfinite(x) else "" for x in values]])
    arrays_atomic(output/"selected.npz", selected)
    json_atomic(output/"status.json", {"status": "complete", "human_changed": changed, "run_id": run_id})
    return HumanRepairResult(selected, changed, report)
