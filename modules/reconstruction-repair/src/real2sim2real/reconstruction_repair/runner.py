"""Atomic, fingerprinted stage continuation and portable output publication."""

from dataclasses import asdict
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from .bundle import (arrays_atomic, arrays_read, asset_path, bundle_read, file_sha256,
                     geometry_validate, json_atomic)
from .contracts import ReconstructionBundle, RepairConfig, RepairProposal, RunResult
from .repairs import PROPOSERS, STAGES
from .scoring import geometry_scores, scores_compare
from .validation import proposal_validate


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _implementation() -> dict:
    package = Path(__file__).parent
    return {"source_sha256": {p.name: file_sha256(p) for p in sorted(package.glob("*.py"))},
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow")}}


def _frame_table(path: Path, before, after) -> None:
    def number(x):
        return float(x) if np.isfinite(x) else ""
    with path.open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["frame_id", "iou_before", "iou_after", "iou_delta", "camera_center_depth_change_m",
                      "max_hand_surface_gap_before_m", "max_hand_surface_gap_after_m"])
        a, b = before.per_frame, after.per_frame
        for i, frame in enumerate(a["frame_ids"]):
            out.writerow([int(frame), number(a["iou"][i]), number(b["iou"][i]), number(b["iou"][i] - a["iou"][i]),
                          number(b["camera_center_m"][i, 2] - a["camera_center_m"][i, 2]),
                          number(a["hand_gap_m"][i].max()) if a["hand_gap_m"].shape[1] else "",
                          number(b["hand_gap_m"][i].max()) if b["hand_gap_m"].shape[1] else ""])


def _review_markdown(report: dict) -> str:
    lines = ["# Initialization repair review", "", f"Sequence: `{report['sequence_id']}`", "",
             f"Result: **{report['status']}**. Human payload preserved: **{report['human_payload_preserved']}**.", "",
             "| Stage | Outcome | Explanation |", "|---|---|---|"]
    for stage in report["stages"]:
        text = ", ".join(stage.get("validation", {}).get("reasons", [])) or stage["reason"]
        lines.append(f"| {stage['stage']} | {stage['outcome']} | {text} |")
    lines += ["", "## Diagnostic scores", "", "| Metric | Before | After | Delta |", "|---|---:|---:|---:|"]
    comparison = report["scores"]
    for name in ("silhouette_iou", "center_acceleration_m_s2", "angular_speed_deg_s", "hand_surface_gap_m", "semantic_facing_agreement"):
        values = [comparison[k][name]["mean"] for k in ("before", "after", "after_minus_before")]
        display = ["unavailable" if v is None else f"{v:.6g}" for v in values]
        lines.append(f"| {name} (mean) | {' | '.join(display)} |")
    lines += ["", "These diagnostics describe geometry and evidence quality; physical task outcomes require separate evaluation.",
              "Lower motion can remove real movement; smaller unsigned hand gaps can hide penetration.",
              "Image overlap does not settle depth, semantic orientation or hand contact.", "", "## Remaining review items", ""]
    lines.extend(f"- {warning}" for warning in report["review_items"])
    human = report.get("human_repair")
    if human is not None:
        lines += ["", "## Native-MHR human diagnostics", "", "| Metric (mean) | Before | After |", "|---|---:|---:|"]
        for metric in ("silhouette_iou", "joint_error_px", "joint_acceleration_m_s2", "sampled_hand_gap_m"):
            a, b = human["scores"]["before"][metric]["mean"], human["scores"]["after"][metric]["mean"]
            lines.append(f"| {metric} | {a} | {b} |")
        lines += ["", "Human detail: `human/report.json`, `human/per_frame.csv`, `human_scores.json`.",
                  "Original hand observations remain input evidence; regenerate derived contacts after an accepted human repair."]
    lines += ["", "No CARI4D refinement or cloud job was launched. Optional human fitting is a separate bounded CPU optimization.",
              "Any changed initialization invalidates old downstream optimizer state.",
              "Use `scores.json` for the complete before/after table and `per_frame.csv` for individual frames.", ""]
    if 'gate_explanation' in report:
        from .gate_report import markdown
        lines += ['', markdown(report['gate_explanation'])]
    return "\n".join(lines)


def repair_run(source: Path, output: Path, config: RepairConfig | None = None, *, resume: bool = False,
               human_decoder=None, metric_calibration=None, metric_evidence=None,
               metric_provenance=None, diagnostic_only=False) -> RunResult:
    config = RepairConfig.from_dict(asdict(config or RepairConfig()))
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and run directories must be separate, non-nested paths")
    bundle = bundle_read(source)
    if bundle.manifest.get("repair", {}).get("original_hand_observations_require_refresh") and any(
            getattr(config, stage) for stage in STAGES if stage != "topology"):
        raise ValueError("Human repair invalidated derived hand observations; rebuild evidence in a new bundle before further object repairs")
    if config.human_repair and human_decoder is None:
        raise ValueError("Opt-in human repair requires a pinned native-MHR decoder")
    if not config.human_repair and human_decoder is not None:
        raise ValueError("A supplied decoder requires human_repair=true")
    if output.exists() and not resume:
        raise ValueError("Output exists; use --resume only for the same input, code and configuration")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another repair process holds the run lock") from exc
        from .metric_preflight import fine_repair_requested, inspect_metric_calibration
        requested = fine_repair_requested(config)
        metric = dict(stage='metric_size_preflight', passed=False, current_mesh_size_verified=False,
                      status='not_required_no_fine_repair', diagnostic_only=bool(diagnostic_only))
        if diagnostic_only:
            if any(x is not None for x in (metric_calibration, metric_evidence, metric_provenance)):
                raise ValueError('Diagnostic bypass cannot also claim metric-size validation')
            metric['status'] = 'explicit_diagnostic_without_metric_validation'
        elif requested:
            try:
                if (metric_evidence is None) != (metric_provenance is None):
                    raise ValueError('Provide both metric evidence and metric provenance')
                if metric_calibration is not None and metric_evidence is not None:
                    raise ValueError('Use either a calibration directory or new metric evidence, not both')
                if metric_evidence is not None:
                    from .metric_calibration import calibrate_object
                    metric_calibration = output / 'metric_calibration'
                    if not metric_calibration.exists():
                        calibrate_object(source, metric_evidence, metric_provenance, metric_calibration)
                    else:
                        if (file_sha256(metric_evidence) != file_sha256(metric_calibration/'evidence.npz')
                                or file_sha256(metric_provenance) != file_sha256(metric_calibration/'evidence_provenance.json')):
                            raise ValueError('Resume metric evidence changed')
                if metric_calibration is None:
                    raise ValueError('Metric-size calibration is required before fine pose repair; supply a calibration or metric evidence/provenance')
                metric = inspect_metric_calibration(bundle, metric_calibration)
                json_atomic(output / 'metric_size_preflight.json', metric)
                if not metric['passed']:
                    advice=('retain the size hypothesis, resize/retrack and calibrate the new bundle before fine repair'
                            if metric['requires_resize_and_fresh_tracking'] else
                            'retain the input mesh and resolve rejected calibration evidence; resizing is not justified')
                    raise ValueError('Fine repair blocked: ' + metric['status'] + '; ' + advice)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                json_atomic(output / 'metric_size_preflight.json', dict(metric, error=str(exc)))
                json_atomic(output / 'status.json', dict(status='blocked_metric_size_preflight', error=str(exc), fine_repair_started=False))
                from .gate_report import write_explanation
                write_explanation(dict(schema='r2s2r_engines.external-comparison-gates/1',gates=[dict(
                    id='metric_size_current_mesh',stage='metric_size',status='not_verified',importance='required_validity',
                    observed=metric.get('scale_multiplier'),explanation=str(exc),
                    recommended_action='Retain calibration evidence; resolve this before normal fine repair.')],
                    components=[dict(stage='fine_pose_repair',outcome='not_run')]),output)
                raise
        json_atomic(output / 'metric_size_preflight.json', metric)
        identity = {"input_manifest_sha256": file_sha256(source / "manifest.json"),
                    "input_assets": bundle.manifest["assets"], "config": asdict(config), "implementation": _implementation(),
                    "metric_size_preflight": metric, "diagnostic_only": bool(diagnostic_only)}
        if human_decoder is not None:
            identity["human_decoder"] = human_decoder.identity
        run_id = _fingerprint(identity)
        identity_path = output / "run.json"
        if identity_path.exists():
            saved = json.loads(identity_path.read_text())
            if saved != {"run_id": run_id, **identity}:
                raise ValueError("Resume mismatch: inputs, configuration, source or dependency versions changed; use a new output")
        else:
            owned = {'run.lock', 'metric_size_preflight.json', 'metric_calibration', 'status.json', 'GATES.md', 'gate_explanation.json'}
            if any(p.name not in owned for p in output.iterdir()):
                raise ValueError("Run identity is missing; do not overwrite an unrecognized directory")
            json_atomic(identity_path, {"run_id": run_id, **identity})
        try:
            return _run_stages(bundle, output, config, run_id, human_decoder, metric, diagnostic_only)
        except BaseException as exc:
            json_atomic(output / "status.json", {"status": "interrupted", "error": str(exc), "run_id": run_id,
                                                  "recovery": "Resume with identical inputs/config/code; completed stages are verified."})
            raise


def _run_stages(bundle: ReconstructionBundle, output: Path, config: RepairConfig, run_id: str,
                human_decoder=None, metric=None, diagnostic_only=False) -> RunResult:
    initial_scores = geometry_scores(bundle, max_gap_s=config.max_gap_s)
    current, current_scores = bundle.geometry, initial_scores
    parent_sha = bundle.manifest["assets"]["geometry"]["sha256"]
    records = []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for index, stage in enumerate(STAGES):
        json_atomic(output / "status.json", {"status": "running", "stage": stage, "completed_stages": index, "run_id": run_id})
        checkpoint = checkpoint_dir / f"{index:02d}_{stage}.npz"
        receipt = checkpoint.with_suffix(".json")
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["run_id"] != run_id or record["parent_sha256"] != parent_sha or record["selected_sha256"] != file_sha256(checkpoint):
                raise ValueError(f"Corrupt or incompatible stage checkpoint: {stage}")
            current = arrays_read(checkpoint)
            geometry_validate(current)
            current_scores = geometry_scores(bundle, current, config.max_gap_s)
        else:
            record = {"run_id": run_id, "stage": stage, "parent_sha256": parent_sha}
            if stage == 'scale' and metric.get('passed'):
                record.update(outcome='disabled', reason='Metric-checked size retained; resizing requires new calibration and fresh tracking')
            elif not getattr(config, stage):
                record.update(outcome="disabled", reason="Disabled in the declared configuration")
            else:
                proposal = PROPOSERS[stage](bundle, current, config)
                same = all(np.array_equal(current[key], proposal.geometry[key]) for key in current)
                record.update(reason=proposal.reason, details=proposal.details)
                if same:
                    record.update(outcome="unchanged")
                else:
                    proposal_scores = geometry_scores(bundle, proposal.geometry, config.max_gap_s)
                    validation = proposal_validate(bundle, current, proposal, current_scores, proposal_scores, config)
                    record.update(validation=validation, scores=scores_compare(current_scores, proposal_scores),
                                  outcome="accepted" if validation["accepted"] else "rejected")
                    if validation["accepted"]:
                        current, current_scores = proposal.geometry, proposal_scores
            arrays_atomic(checkpoint, current)
            record["selected_sha256"] = file_sha256(checkpoint)
            json_atomic(receipt, record)  # receipt is the completed-stage commit marker
        parent_sha = record["selected_sha256"]
        records.append(record)
    # Stage-wise tolerances cannot accumulate into an unreviewed regression.
    # Check the complete selection against the original input as well.
    aggregate = None
    if any(r["outcome"] == "accepted" and r["stage"] != "topology" for r in records):
        aggregate = proposal_validate(bundle, bundle.geometry, RepairProposal("combined", current, "Combined repairs"),
                                      initial_scores, current_scores, config)
        if not aggregate["accepted"]:
            current = bundle.geometry
            current_scores = initial_scores
            # Keep checkpoints as proposal history; explicitly mark all earlier
            # accepted decisions as superseded in the final report.
            records = [{**r, "outcome": "superseded_by_final_fallback"} if r["outcome"] == "accepted" else r for r in records]
    human = None
    if config.human_repair:
        from .human_runner import human_stage_run
        json_atomic(output / "status.json", {"status": "running", "stage": "human_repair", "run_id": run_id})
        human = human_stage_run(bundle, current, output / "human", config, human_decoder, run_id)
        if human.report["requires_object_fallback"]:
            current, current_scores = bundle.geometry, initial_scores
            records = [{**r, "outcome": "superseded_by_pair_fallback"} if r["outcome"] == "accepted" else r for r in records]
        records.extend(human.report["stages"])
    # Verify original source again before publishing any portable bundle.
    checked = bundle_read(bundle.root)
    if checked.manifest != bundle.manifest:
        raise ValueError("Input manifest changed while repairs were running")
    geometry_changed = any(not np.array_equal(bundle.geometry[key], current[key]) for key in current)
    human_changed = human is not None and human.changed
    changed = geometry_changed or human_changed
    review = ["Metric depth and hand contact remain uncertain without independent reference evidence.",
              "Human identity, finger controls and root orientation are preserved; depth/contact still require independent evidence."]
    if not config.human_repair:
        review.append("Optional native-MHR repair is disabled; the supplied human payload is preserved.")
    if human is not None and human.report["requires_object_fallback"]:
        review.append("Combined decoded-human/object separation exceeded its bound; both original components were restored.")
    mesh = current_scores.summary["mesh"]
    if mesh["nonmanifold_edges"]:
        review.append("Mesh still has nonmanifold edges; topology repair needs inspection.")
    if mesh["vertices"] > config.mesh_vertex_budget or mesh["faces"] > config.mesh_face_budget:
        review.append("Mesh exceeds the declared downstream packing budget; no automatic destructive decimation was applied.")
    if current_scores.summary["image_reviewed_frames"] < len(current["frame_ids"]):
        review.append("Some frames lack reliable visible masks or valid projections; their poses are withheld from automatic repair.")
    if "hand_points_world" not in bundle.observations:
        review.append("Hand-distance evidence is absent; contact is unverified.")
    if any(x["outcome"] == "rejected" for x in records):
        review.append("One or more proposals failed regression gates; their parent reconstruction was retained.")
    if aggregate is not None and not aggregate["accepted"]:
        review.append("Combined repairs failed the final comparison with the original input; the original initialization was restored.")
    report = {"schema": "r2s2r_engines.repair-report/1", "run_id": run_id, "sequence_id": bundle.manifest["sequence_id"],
              "run_identity": json.loads((output / "run.json").read_text()),
              "status": "repaired_with_review_items" if changed else "unchanged_with_review_items",
              "human_payload_preserved": not human_changed, "human_payload_present": "human" in bundle.manifest["assets"], "stages": records,
              "human_repair": human.report if human is not None else None,
              "aggregate_validation": aggregate,
              "metric_size_preflight": metric, "diagnostic_only": bool(diagnostic_only),
              "metric_size_verified": bool(metric.get('passed')), "approved_for_use": False,
              "scores": scores_compare(initial_scores, current_scores), "review_items": review,
              "requires_optimizer_restart": changed, "optimizer_start_update": 0 if changed else None,
              "invalidated_downstream": ["object-derived contact estimates", "object point samples", "optimizer states",
                                         "cached renderings and exports"] if changed else [],
              "cloud_launch_authorized": False, "output_bundle": "bundle"}
    if human_changed:
        report["scores"]["interpretation"]["human"] = "Human changed; see human_scores.json. Object-stage hand diagnostics use original input observations."
        report["invalidated_downstream"].append("human-derived hand/contact observations")
    from .gate_report import write_explanation
    report['gate_explanation'] = write_explanation(report, output)
    json_atomic(output / "report.json", report)
    json_atomic(output / "scores.json", report["scores"])
    if human is not None:
        json_atomic(output / "human_scores.json", human.report["scores"])
    _frame_table(output / "per_frame.csv", initial_scores, current_scores)
    (output / "REVIEW.md").write_text(_review_markdown(report))
    # Build privately, validate completely, then rename into place. On resume,
    # a previously completed output must have valid checksums before replacement.
    destination, staging = output / "bundle", output / ".bundle-staging"
    if destination.exists():
        existing = bundle_read(destination)
        if existing.manifest.get("repair", {}).get("run_id") != run_id:
            raise ValueError("Existing output bundle belongs to a different run")
        # Completed output is immutable. The newly derived geometry must agree.
        if any(not np.array_equal(existing.geometry[key], current[key]) for key in current):
            raise ValueError("Completed output no longer matches verified checkpoints")
        if human_changed:
            old_human = arrays_read(asset_path(existing.root, existing.manifest["assets"]["human"]))
            if set(old_human) != set(human.arrays) or any(not np.array_equal(old_human[k], human.arrays[k]) for k in old_human):
                raise ValueError("Completed human output no longer matches verified checkpoints")
    else:
        if staging.exists():
            shutil.rmtree(staging)  # only this pipeline's incomplete publication area
        staging.mkdir()
        manifest = {**bundle.manifest, "assets": {}, "repair": {"run_id": run_id, "parent_manifest_sha256": file_sha256(bundle.root / "manifest.json"),
                     "metric_size_verified": bool(metric.get('passed')), "metric_size_preflight": metric,
                     "diagnostic_only": bool(diagnostic_only),
                     "human_applied": human_changed or bundle.manifest.get("repair", {}).get("human_applied", False),
                     "original_hand_observations_require_refresh": human_changed or bundle.manifest.get("repair", {}).get("original_hand_observations_require_refresh", False)}}
        if human_changed:
            manifest["human"] = {**bundle.manifest["human"],
                                 "parent_parity_source": bundle.manifest["human"]["parity_source"],
                                 "parity_source": f"Verified native-MHR decoder re-export after repair {run_id}"}
        if "baseline_human" in bundle.manifest["assets"] and not human_changed:
            baseline = staging / "baseline_human.npz"
            shutil.copyfile(asset_path(bundle.root, bundle.manifest["assets"]["baseline_human"]), baseline)
            manifest["assets"]["baseline_human"] = {"path": baseline.name, "sha256": file_sha256(baseline)}
        for role in ("human", "observations"):
            if role in bundle.manifest["assets"]:
                target = staging / f"{role}.npz"
                if role == "human" and human_changed:
                    arrays_atomic(target, human.arrays)
                    baseline = staging / "baseline_human.npz"
                    shutil.copyfile(asset_path(bundle.root, bundle.manifest["assets"][role]), baseline)
                    manifest["assets"]["baseline_human"] = {"path": baseline.name, "sha256": file_sha256(baseline)}
                else:
                    shutil.copyfile(asset_path(bundle.root, bundle.manifest["assets"][role]), target)
                manifest["assets"][role] = {"path": target.name, "sha256": file_sha256(target)}
        geometry_path = staging / "geometry.npz"
        if geometry_changed:
            arrays_atomic(geometry_path, current)
        else:
            shutil.copyfile(asset_path(bundle.root, bundle.manifest["assets"]["geometry"]), geometry_path)
        shutil.copyfile(asset_path(bundle.root, bundle.manifest["assets"]["geometry"]), staging / "baseline_geometry.npz")
        shutil.copyfile(output / "report.json", staging / "repair_report.json")
        for role in ("geometry", "baseline_geometry", "repair_report"):
            target = staging / (f"{role}.json" if role == "repair_report" else f"{role}.npz")
            manifest["assets"][role] = {"path": target.name, "sha256": file_sha256(target)}
        json_atomic(staging / "manifest.json", manifest)
        bundle_read(staging)
        staging.rename(destination)
    json_atomic(output / "status.json", {"status": "complete", "run_id": run_id, "output_bundle": "bundle",
                                          "accepted_stages": [r["stage"] for r in records if r["outcome"] == "accepted"]})
    return RunResult(destination, report)
