"""Evidence audit before robot-specific retargeting or control experiments."""

import json
from pathlib import Path
import numpy as np

from real2sim2real.artifacts import digest, write_json
from .hand_audit import intervals


def coverage_assessment(accepted, detected, candidates, frame_ids, times):
    accepted, detected, candidates = [
        np.asarray(a, bool) for a in (accepted, detected, candidates)
    ]
    if (
        accepted.ndim != 2
        or accepted.shape[1] != 2
        or not accepted.shape == detected.shape == candidates.shape
    ):
        raise ValueError("Expected matching Nx2 coverage masks")
    if np.any(accepted & ~detected) or np.any(candidates & detected):
        raise ValueError(
            "Metric estimates require detections; probe candidates must be new"
        )
    rows = {}
    for h, name in enumerate(("left", "right")):
        rows[name] = dict(
            accepted_metric_frames=int(accepted[:, h].sum()),
            missing_metric_frames=int((~accepted[:, h]).sum()),
            new_2d_candidates=int(candidates[:, h].sum()),
            missing_2d_after_probe=int((~(detected[:, h] | candidates[:, h])).sum()),
            remaining_metric_gaps_even_if_all_new_2d_candidates_pass_3d=int(
                (~(accepted[:, h] | candidates[:, h])).sum()
            ),
            missing_metric_intervals=intervals(~accepted[:, h], frame_ids, times),
            still_missing_2d_intervals=intervals(
                ~(detected[:, h] | candidates[:, h]), frame_ids, times
            ),
        )
    return dict(
        total_frames=len(accepted),
        hands=rows,
        complete_metric_hands=bool(accepted.all()),
        both_hands_accepted_intervals=intervals(accepted.all(1), frame_ids, times),
        intervals_are_not_control_approvals=True,
        decision="not_ready_for_full_sequence_control"
        if not accepted.all()
        else "requires_robot_specific_validation",
        control_ready=False,
    )


def assess(config):
    root = Path(config["output"])
    output = root / "readiness_assessment.json"
    if output.exists():
        raise FileExistsError("Preserve existing readiness assessment")
    hands = np.load(root / "hands_metric/hands.npz")
    observations = np.load(root / "hands2d/landmarks.npz")
    quality = json.loads((root / "hands_metric/quality.json").read_text())
    summary = json.loads((root / "quality_summary.json").read_text())
    if summary["sources"]["hands_metric/hands.npz"] != digest(
        root / "hands_metric/hands.npz"
    ):
        raise ValueError("Readiness summary describes different hands")
    candidates = np.zeros_like(hands["accepted"])
    sources = {
        name: digest(root / name)
        for name in (
            "hands_metric/hands.npz",
            "hands2d/landmarks.npz",
            "quality_summary.json",
        )
    }
    probe = config.get("observation_probe_path")
    if probe:
        probe = Path(probe)
        data = np.load(probe / "candidates.npz")
        report = json.loads((probe / "summary.json").read_text())
        if report["parent_artifact_sha256"] != quality.get("temporal_parent_sha256"):
            raise ValueError("Probe and temporal fit must share their source parent")
        if not np.array_equal(
            data["frame_ids"], hands["frame_ids"]
        ) or not np.array_equal(data["timestamps_s"], hands["timestamps_s"]):
            raise ValueError("Probe and metric hands must share the exact timeline")
        candidates = data["accepted"]
        sources.update(
            probe_candidates=digest(probe / "candidates.npz"),
            probe_summary=digest(probe / "summary.json"),
        )
    result = coverage_assessment(
        hands["accepted"],
        observations["accepted"],
        candidates,
        hands["frame_ids"],
        hands["timestamps_s"],
    )
    result.update(
        schema="real2sim2real.interaction-readiness/1",
        sources=sources,
        contact_findings={
            name: {
                key: row[key]
                for key in (
                    "object_penetration_frames_over_tolerance",
                    "table_penetration_frames_over_tolerance",
                )
            }
            for name, row in summary["hands"].items()
        },
        object_table_max_penetration_m=summary["object_table_max_penetration_m"],
        requirements_not_validated=[
            "Joint visibility and occlusion uncertainty",
            "Robot-specific kinematics and retargeting",
            "Dynamic contact replay and controller behavior",
        ],
        conclusion="Current observations do not support a complete two-hand control reference. Temporal priors and crop retries cannot replace missing measured evidence."
        if not hands["accepted"].all()
        else "Geometric completeness alone does not establish suitability for a particular robot or controller.",
        candidate_note="New 2D candidates have not been promoted to metric hands. The hypothetical coverage bound assumes all would pass 3D checks; it is not a measured result.",
    )
    write_json(output, result)
    return result
