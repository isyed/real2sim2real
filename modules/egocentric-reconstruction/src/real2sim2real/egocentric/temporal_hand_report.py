"""Paired temporal metrics and explicit missing-evidence control blockers."""

import json
from pathlib import Path
import numpy as np

from real2sim2real.artifacts import digest, load, write_json
from .hand_audit import intervals
from .hand_temporal import temporal_measurements
from .interaction_summary import statistics


def compare(config):
    parent, root = Path(config["hand_temporal"]["parent"]), Path(config["output"])
    output = root / "temporal_comparison.json"
    if output.exists():
        raise FileExistsError("Preserve existing temporal comparison")
    load(parent, kinds={"metric-interaction-reconstruction"})
    before, after = [np.load(p / "hands_metric/hands.npz") for p in (parent, root)]
    q0, q1 = [
        json.loads((p / "hands_metric/quality.json").read_text())
        for p in (parent, root)
    ]
    if (
        q1["temporal_parent_sha256"] != digest(parent / "artifact.json")
        or q0["limits"] != q1["limits"]
    ):
        raise ValueError("Temporal parent or acceptance limits differ")
    for key in ("frame_ids", "timestamps_s", "scale_per_hand", "accepted"):
        if not np.array_equal(before[key], after[key]):
            raise ValueError("Temporal fitting changed coverage, timeline or scale")
    changed = after["temporal_refinement_accepted"]
    if np.any(changed & ~before["refinement_accepted"]):
        raise ValueError("Temporal fitting changed an original anchor")
    for key in ("camera_vertices", "camera_joints", "world_vertices", "world_joints"):
        if not np.array_equal(
            before[key][~changed], after[key][~changed], equal_nan=True
        ):
            raise ValueError("Temporal fitting changed undeclared geometry")
    for name in (
        "support/scene.npz",
        "support/support.json",
        "support/tabletop.glb",
        "hands_raw/predictions.npz",
        "hands2d/landmarks.npz",
    ):
        if digest(parent / name) != digest(root / name):
            raise ValueError("Temporal fitting changed fixed inputs: " + name)
    contacts = json.loads((root / "contacts/diagnostics.json").read_text())
    if contacts["summary"]["sources"]["hands"] != digest(
        root / "hands_metric/hands.npz"
    ):
        raise ValueError("Temporal contacts describe different hands")
    maximum_step = q1["temporal_options"]["max_step_s"]
    motion = [
        temporal_measurements(
            h["world_joints"], h["accepted"], h["timestamps_s"], max_step_s=maximum_step
        )
        for h in (before, after)
    ]
    report = {}
    for h, name in enumerate(("left", "right")):
        ids = np.flatnonzero(changed[:, h])
        edges, triples = motion[0][2][:, h], motion[0][3][:, h]
        affected = triples & (changed[:-2, h] | changed[1:-1, h] | changed[2:, h])
        metrics = {}
        for label, (velocity, acceleration, _, _) in zip(
            ("parent", "temporal"), motion
        ):
            magnitudes = np.linalg.norm(acceleration[:, h], axis=-1)
            metrics[label] = dict(
                joint_speed_m_s=statistics(
                    np.linalg.norm(velocity[:, h][edges], axis=-1).ravel()
                ),
                joint_acceleration_m_s2=statistics(magnitudes[triples].ravel()),
                changed_interval_acceleration_m_s2=statistics(
                    magnitudes[affected].ravel()
                ),
                changed_interval_acceleration_rms_m_s2=float(
                    np.sqrt(np.mean(magnitudes[affected] ** 2))
                )
                if affected.any()
                else None,
            )
        report[name] = dict(
            accepted_frames=int(after["accepted"][:, h].sum()),
            changed_frame_ids=after["frame_ids"][ids].tolist(),
            remaining_intervals=intervals(
                ~after["accepted"][:, h], after["frame_ids"], after["timestamps_s"]
            ),
            motion=metrics,
            changed_frame_fit_metrics={
                key: {
                    label: statistics([q["records"][i][h][key] for i in ids])
                    for label, q in (("parent", q0), ("temporal", q1))
                }
                for key in (
                    "reprojection_median_px",
                    "reprojection_p90_px",
                    "validation_surface_median_m",
                    "validation_surface_p90_m",
                )
            },
        )
    summary = json.loads((root / "quality_summary.json").read_text())
    blockers = []
    for name in ("left", "right"):
        row = summary["hands"][name]
        if row["accepted_metric_frames"] < summary["total_frames"]:
            blockers.append(
                dict(
                    hand=name,
                    reason="missing_accepted_metric_hands",
                    frames=summary["total_frames"] - row["accepted_metric_frames"],
                )
            )
        if (
            row["object_penetration_frames_over_tolerance"]
            or row["table_penetration_frames_over_tolerance"]
        ):
            blockers.append(dict(hand=name, reason="penetration_over_tolerance"))
    result = dict(
        schema="real2sim2real.temporal-hand-comparison/1",
        total_frames=len(changed),
        hands=report,
        coverage_unchanged=True,
        original_unrefined_anchors_unchanged=True,
        fixed_scene_shape_scale_observations_unchanged=True,
        control_ready=False,
        observed_blockers=blockers,
        unvalidated_requirements=[
            "Occluded joint uncertainty is not calibrated.",
            "Robot-specific retargeting, collision and dynamic replay have not been validated.",
        ],
        note="Temporal coherence and fit residuals do not establish physical correctness. No missing poses were synthesized.",
        sources=dict(
            parent_artifact=digest(parent / "artifact.json"),
            candidate_hands=digest(root / "hands_metric/hands.npz"),
            candidate_contacts=digest(root / "contacts/diagnostics.json"),
        ),
    )
    write_json(output, result)
    return result
