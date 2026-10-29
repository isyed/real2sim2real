"""Explicit adapters; no historical path discovery, GPU inference or submission."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from .bundle import arrays_read, bundle_copy, bundle_create, bundle_read, file_sha256, json_atomic
from .contracts import BundleData, RepairConfig
from .runner import repair_run
from .scoring import geometry_scores, reference_scores, scores_compare


def _import(args):
    raw, camera = arrays_read(args.source), arrays_read(args.camera)
    for old, new in (("object_vertices", "vertices"), ("object_faces", "faces")):
        if old in raw:
            if new in raw:
                raise ValueError(f"Ambiguous mesh fields: both {old} and {new}")
            raw[new] = raw.pop(old)
    required = {"vertices", "faces", "object_rotation", "object_translation", "object_scale", "frames"}
    if required - set(raw):
        raise ValueError(f"Source archive missing {sorted(required - set(raw))}")
    metadata = json.loads(args.metadata.read_text())
    n = len(raw["frames"])
    geometry = {key: raw[key] for key in required if key != "frames"}
    geometry["frame_ids"] = raw["frames"]
    if args.fps is not None:
        if not np.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("fps must be finite and positive")
        geometry["timestamps_s"] = raw["frames"].astype(float) / args.fps
    else:
        geometry["timestamps_s"] = np.load(args.timestamps, allow_pickle=False)
    for key, shape in (("intrinsics", (3, 3)), ("world_to_camera", (4, 4))):
        value = camera[key]
        geometry[key] = np.broadcast_to(value, (n, *shape)).copy() if value.shape == shape else value
    if "frame_ids" in camera and not np.array_equal(camera["frame_ids"], geometry["frame_ids"]):
        raise ValueError("Camera frame IDs do not match source")
    extras = set(raw) - required
    if extras - {"pose", "shape", "scales"}:
        raise ValueError(f"Unrecognized source fields must be explicitly mapped, not silently discarded: {sorted(extras)}")
    human = {key: raw[key] for key in extras} or None
    if human:
        for key, width in (("pose", 136), ("shape", 45), ("scales", 68)):
            allowed = [(n, width)] if key == "pose" else [(width,), (n, width)]
            if key not in human or human[key].shape not in allowed:
                raise ValueError(f"Native MHR import requires {key} with shape in {allowed}")
    observations = arrays_read(args.observations) if args.observations else None
    metadata["import_provenance"] = {"source_sha256": file_sha256(args.source), "camera_sha256": file_sha256(args.camera),
                                    "timestamps": "frame_ids / explicit fps" if args.fps else "explicit timestamps_s",
                                    "fps": args.fps}
    bundle = bundle_create(args.output, BundleData(metadata, geometry, observations, human))
    return {"bundle": str(bundle.root), "frames": n}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair existing reconstruction initializations with portable evidence and before/after diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Verify schema, numeric conventions and every asset hash")
    validate.add_argument("bundle", type=Path)
    config = sub.add_parser("default-config", help="Print the declared default repair policy")
    run = sub.add_parser("run", help="Repair and score; preserve parent for rejected proposals")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--metric-calibration", type=Path, help="Retained metric calibration for this exact current input bundle")
    run.add_argument("--metric-evidence", type=Path, help="Run required calibration before fine repair")
    run.add_argument("--metric-provenance", type=Path)
    run.add_argument("--diagnostic-without-metric-calibration", action="store_true",
                     help="Explicit unverified diagnostic only; never claims metric-size validation")
    run.add_argument("--mhr-decoder", type=Path, help="Pinned asset specification for opt-in human repair, inside a CPU torch runtime")
    export = sub.add_parser("export", help="Create a verified independent copy of a portable bundle")
    export.add_argument("--input", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    legacy = sub.add_parser("import-npz", help="Import existing source workflow-style numeric archives with explicit cameras/conventions")
    for name in ("source", "camera", "metadata", "output"):
        legacy.add_argument(f"--{name}", required=True, type=Path)
    legacy.add_argument("--observations", type=Path)
    time = legacy.add_mutually_exclusive_group(required=True)
    time.add_argument("--fps", type=float)
    time.add_argument("--timestamps", type=Path)
    score = sub.add_parser("score", help="Compare any compatible before/after bundles, optionally against a reference")
    for name in ("before", "after", "output"):
        score.add_argument(f"--{name}", required=True, type=Path)
    score.add_argument("--reference", type=Path)
    score.add_argument("--surface-samples", type=int, default=2048)
    calibration = sub.add_parser("calibrate-object", help="Diagnose metric size and produce a mesh hypothesis requiring fresh tracking")
    for name in ("input", "evidence", "provenance", "output"):
        calibration.add_argument("--"+name, type=Path, required=True)
    temporal = sub.add_parser("refine-object-window", help="Fit a bounded multi-frame object-pose hypothesis on CPU")
    for name in ("input", "provenance", "output"):
        temporal.add_argument("--"+name, type=Path, required=True)
    temporal.add_argument("--config",type=Path)
    temporal.add_argument("--metric-bundle", type=Path)
    temporal.add_argument("--metric-calibration", type=Path)
    temporal.add_argument("--diagnostic-without-metric-calibration", action="store_true")
    seed = sub.add_parser("register-object-seed", help="Fresh CPU translation seed after verified size calibration; not a trajectory")
    for name in ("calibration", "input", "provenance", "output"):
        seed.add_argument("--" + name, type=Path, required=True)
    explanation = sub.add_parser('explain-comparison', help='Explain passed/failed/unknown gates without changing acceptance')
    explanation.add_argument('--report', type=Path, required=True)
    explanation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            b = bundle_read(args.bundle)
            result = {"valid": True, "sequence_id": b.manifest["sequence_id"], "frames": len(b.geometry["frame_ids"])}
        elif args.command == "default-config":
            result = asdict(RepairConfig())
        elif args.command == "run":
            cfg = RepairConfig.from_dict(json.loads(args.config.read_text()) if args.config else {})
            decoder = None
            if args.mhr_decoder:
                from .human_decoder import TorchMHRDecoder
                try:
                    decoder = TorchMHRDecoder(args.mhr_decoder)
                except ImportError as exc:
                    raise ValueError("Native human repair needs the optional CPU torch runtime; keep the base host environment lightweight") from exc
            r = repair_run(args.input, args.output, cfg, resume=args.resume, human_decoder=decoder,
                metric_calibration=args.metric_calibration, metric_evidence=args.metric_evidence,
                metric_provenance=args.metric_provenance, diagnostic_only=args.diagnostic_without_metric_calibration)
            result = {"bundle": str(r.output_bundle), "status": r.report["status"],
                      "accepted": [s["stage"] for s in r.report["stages"] if s["outcome"] == "accepted"]}
        elif args.command == "export":
            result = {"bundle": str(bundle_copy(args.input, args.output).root)}
        elif args.command == "import-npz":
            result = _import(args)
        elif args.command == "calibrate-object":
            from .metric_calibration import calibrate_object
            report = calibrate_object(args.input,args.evidence,args.provenance,args.output)
            result = {k:report[k] for k in ("status","scale_multiplier","requires_retracking","approved_for_use","reasons")}
        elif args.command == "refine-object-window":
            from .temporal_pose import run_window,TemporalPoseConfig
            try:cfg=TemporalPoseConfig(**json.loads(args.config.read_text())) if args.config else TemporalPoseConfig()
            except TypeError as exc:raise ValueError('Unknown temporal configuration field or invalid configuration object') from exc
            report=run_window(args.input,args.provenance,args.output,cfg,
                metric_bundle=args.metric_bundle,metric_calibration=args.metric_calibration,
                diagnostic_only=args.diagnostic_without_metric_calibration)
            result={"window_passed":report["decision"]["accepted"],"reasons":report["decision"]["reasons"],
                    "full_trajectory_accepted":False,"approved_for_use":False}
        elif args.command == "register-object-seed":
            from .metric_seed import seed_from_calibration
            result = seed_from_calibration(args.calibration, args.input, args.provenance, args.output)
        elif args.command == 'explain-comparison':
            from .gate_report import explain_file
            report = explain_file(args.report, args.output)
            result = dict(output=str(args.output), required_checks_unresolved=report['required_checks_unresolved'],
                          actor_training_authorized=False, acceptance_policy_changed=False)
        else:
            before, after = bundle_read(args.before), bundle_read(args.after)
            for key in ("frame_ids", "timestamps_s", "intrinsics", "world_to_camera"):
                if not np.array_equal(before.geometry[key], after.geometry[key]):
                    raise ValueError(f"Before/after {key} differs; comparison would be confounded")
            if before.manifest.get("world_frame_id") != after.manifest.get("world_frame_id"):
                raise ValueError("Before/after world frames differ")
            if before.manifest["assets"].get("observations") != after.manifest["assets"].get("observations"):
                # Paths can differ after export; compare content, not path names.
                a = before.manifest["assets"].get("observations", {}).get("sha256")
                b = after.manifest["assets"].get("observations", {}).get("sha256")
                if a != b:
                    raise ValueError("Before/after observations differ; use one fixed evaluation set")
            result = scores_compare(geometry_scores(before), geometry_scores(after))
            from .gate_report import explain
            if 'repair_report' in after.manifest['assets']:
                from .bundle import asset_path
                producing = json.loads(asset_path(after.root,after.manifest['assets']['repair_report']).read_text())
                result['gate_explanation'] = explain(producing)
                result['gate_scope'] = dict(source='After-bundle producing repair report',
                    before_is_producing_parent=producing.get('run_identity',{}).get('input_manifest_sha256')==file_sha256(before.root/'manifest.json'))
            else:
                result['gate_explanation'] = explain(dict(schema='r2s2r_engines.external-comparison-gates/1',gates=[dict(
                    id='metric_size_current_mesh',stage='metric_size',status='not_checked',importance='required_validity',
                    explanation='The supplied bundles have no producing repair gate report.',
                    recommended_action='Do not infer metric-size or physical readiness from score deltas alone.')]))
            human_same = before.manifest["assets"].get("human", {}).get("sha256") == after.manifest["assets"].get("human", {}).get("sha256")
            result["human_payload_preserved"] = human_same
            if not human_same:
                result["interpretation"]["human"] = "Human payloads differ; these object diagnostics use original hand observations. See the producing human repair report or separate reference diagnostics."
            if args.reference:
                reference = bundle_read(args.reference)
                result["reference"] = {
                    "before": reference_scores(before, reference, args.surface_samples),
                    "after": reference_scores(after, reference, args.surface_samples),
                    "reference_manifest_sha256": file_sha256(reference.root / "manifest.json"),
                }
                ref = result["reference"]
                ref["after_minus_before"] = {
                    key: {stat: ref["after"][key][stat] - value for stat, value in a.items()}
                    for key, a in ref["before"].items() if isinstance(a, dict) and "mean" in a
                }
            if args.output.exists():
                raise ValueError("Score output already exists")
            json_atomic(args.output, result)
            result = {"scores": str(args.output)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
