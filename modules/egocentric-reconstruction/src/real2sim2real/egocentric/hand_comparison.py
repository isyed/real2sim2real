"""Paired refinement evidence with explicit coverage and unchanged parent anchors."""

import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, load, write_json
from .camera import transform
from .hand_audit import intervals
from .interaction_summary import statistics


def compare(parent, candidate):
    parent, candidate = Path(parent), Path(candidate)
    output = candidate / "hand_comparison.json"
    if output.exists():
        raise FileExistsError("Preserve existing hand comparison")
    load(parent, kinds={"metric-interaction-reconstruction"})
    before, after = [
        np.load(root / "hands_metric/hands.npz") for root in (parent, candidate)
    ]
    q0, q1 = [
        json.loads((root / "hands_metric/quality.json").read_text())
        for root in (parent, candidate)
    ]
    if q1.get("refinement_parent_sha256") != digest(parent / "artifact.json"):
        raise ValueError("Candidate does not identify this refinement parent")
    if (
        not np.array_equal(before["frame_ids"], after["frame_ids"])
        or not np.array_equal(before["timestamps_s"], after["timestamps_s"])
        or q0["limits"] != q1["limits"]
        or not np.array_equal(before["scale_per_hand"], after["scale_per_hand"])
    ):
        raise ValueError("Refinement changed timeline, scale or acceptance limits")
    preserved = before["accepted"]
    if not after["accepted"][preserved].all():
        raise ValueError("Refinement lost an accepted parent frame")
    for key in ("camera_vertices", "camera_joints", "world_vertices", "world_joints"):
        if not np.array_equal(before[key][preserved], after[key][preserved]):
            raise ValueError("Refinement changed accepted parent geometry")
    for name in (
        "support/scene.npz",
        "support/support.json",
        "support/tabletop.glb",
        "hands_raw/predictions.npz",
        "hands2d/landmarks.npz",
    ):
        if digest(parent / name) != digest(candidate / name):
            raise ValueError("Refinement changed a fixed source: " + name)
    contacts = json.loads((candidate / "contacts/diagnostics.json").read_text())
    if contacts["summary"]["sources"]["hands"] != digest(
        candidate / "hands_metric/hands.npz"
    ):
        raise ValueError("Comparison contact diagnostics describe different hands")
    scene = np.load(candidate / "support/scene.npz")
    proposed_world = np.full_like(before["proposed_camera_joints"], np.nan)
    for i in range(len(before["frame_ids"])):
        proposed_world[i] = transform(
            before["proposed_camera_joints"][i], scene["world_from_camera"][i]
        )
    report = {}
    for h, name in enumerate(("left", "right")):
        recovered = after["accepted"][:, h] & ~preserved[:, h]
        ids = np.flatnonzero(recovered)
        metrics = {
            key: dict(
                parent_proposals=statistics([q0["records"][i][h][key] for i in ids]),
                refined=statistics([q1["records"][i][h][key] for i in ids]),
            )
            for key in (
                "reprojection_median_px",
                "reprojection_p90_px",
                "validation_surface_median_m",
                "validation_surface_p90_m",
            )
        }
        # Inspect transitions introduced by filling a rejected interval. The
        # baseline side uses rejected proposals, not fabricated accepted motion.
        edges = after["accepted"][1:, h] & after["accepted"][:-1, h]
        edges &= ~(preserved[1:, h] & preserved[:-1, h])
        edges &= np.isfinite(proposed_world[1:, h]).all((1, 2)) & np.isfinite(
            proposed_world[:-1, h]
        ).all((1, 2))
        dt = np.diff(after["timestamps_s"])[edges, None]
        speeds = {}
        for label, joints in (
            ("parent_proposals", proposed_world),
            ("refined", after["world_joints"]),
        ):
            speed = np.linalg.norm(np.diff(joints[:, h], axis=0)[edges], axis=-1) / dt
            speeds[label] = statistics(speed.ravel().tolist())
        report[name] = dict(
            baseline_accepted_frames=int(preserved[:, h].sum()),
            refined_accepted_frames=int(after["accepted"][:, h].sum()),
            recovered_frames=len(ids),
            recovered_frame_ids=after["frame_ids"][ids].tolist(),
            remaining_intervals=intervals(
                ~after["accepted"][:, h], after["frame_ids"], after["timestamps_s"]
            ),
            recovered_fit_metrics=metrics,
            new_temporal_edges=int(edges.sum()),
            new_edge_joint_speed_m_s=speeds,
            recovered_object_penetration_m=statistics(
                [
                    contacts["records"][i]["hands"][h]["object"]["penetration_depth_m"]
                    for i in ids
                ]
            ),
        )
    result = dict(
        schema="real2sim2real.hand-refinement-comparison/1",
        total_frames=len(preserved),
        hands=report,
        accepted_parent_geometry_unchanged=True,
        shape_scale_scene_observations_unchanged=True,
        acceptance_limits_unchanged=True,
        image_metrics_are_fit_residuals=True,
        depth_validation_is_disjoint=True,
        ground_truth_hand_accuracy_established=False,
        control_ready=False,
        sources={
            "parent_artifact": digest(parent / "artifact.json"),
            "candidate_hands": digest(candidate / "hands_metric/hands.npz"),
            "candidate_quality": digest(candidate / "hands_metric/quality.json"),
            "candidate_contacts": digest(candidate / "contacts/diagnostics.json"),
        },
    )
    write_json(output, result)
    return result
