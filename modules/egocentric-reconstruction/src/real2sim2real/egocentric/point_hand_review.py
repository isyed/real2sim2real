"""Evaluation-only metrics and images for a point-guided articulation trial."""

import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real2sim2real.artifacts import digest, write_json
from .hand_experiment_review import draw_projection, distribution
from .hand_observation_fit import visible_landmarks
from .hand_review_inputs import annotation_errors
from .hand_temporal import temporal_measurements
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import project


def boundary_diagnostics(joints, baseline, supported, frame_ids, times):
    """Measure finger motion at baseline/recovery seams; wrist speed is insufficient."""
    joints, baseline, supported = map(np.asarray, (joints, baseline, supported))
    result = []
    for h in range(2):
        for i in np.flatnonzero(baseline[1:, h] != baseline[:-1, h]):
            if not supported[i : i + 2, h].all():
                continue
            if not np.isfinite(joints[i : i + 2, h]).all():
                raise ValueError("Supported boundary joints must be finite")
            dt = float(times[i + 1] - times[i])
            if dt <= 0:
                raise ValueError("Boundary timestamps must increase")
            distances = np.linalg.norm(joints[i + 1, h] - joints[i, h], axis=-1)
            worst = int(distances.argmax())
            result.append(
                dict(
                    hand=h,
                    from_frame=int(frame_ids[i]),
                    to_frame=int(frame_ids[i + 1]),
                    elapsed_s=dt,
                    worst_joint=worst,
                    max_joint_step_m=float(distances[worst]),
                    max_joint_speed_m_s=float(distances[worst] / dt),
                    wrist_step_m=float(distances[0]),
                )
            )
    return result


def diagnostic_support(candidate, baseline):
    tested = candidate.get("sequence_selected", candidate["refinement_attempted"])
    return np.where(
        tested, candidate["diagnostic_geometry_supported"], baseline["accepted"]
    )


def evaluate(config):
    root, parent = Path(config["output"]), Path(config["parent"])
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    receipt = json.loads((root / "prepared.json").read_text())
    for name, sha in receipt["source_files"].items():
        if digest(name) != sha:
            raise ValueError("Baseline source changed: " + name)
    baseline = dict(np.load(root / "fit_inputs/baseline.npz"))
    candidate_dir = root / config.get("candidate_directory", "refined") / "hands_metric"
    candidate = dict(np.load(candidate_dir / "hypotheses.npz"))
    previous = dict(np.load(parent / "hawor/hands_metric/hypotheses.npz"))
    observations = dict(np.load(root / "tracking/point_tracks/observations.npz"))
    quality = json.loads((candidate_dir / "quality.json").read_text())
    immutable = baseline["accepted"] & ~candidate.get(
        "sequence_admitted", np.zeros_like(baseline["accepted"])
    )
    for name in [
        "camera_vertices",
        "camera_joints",
        "world_vertices",
        "world_joints",
        "fits",
    ]:
        if not np.array_equal(
            candidate[name][immutable],
            baseline[name][immutable],
            equal_nan=True,
        ):
            raise ValueError("An accepted baseline pose was modified")
    views_dir = root / "evaluation_inputs/views"
    views = json.loads((views_dir / "views.json").read_text())["views"]
    annotations_file = views_dir / "annotations.json"
    annotations = (
        json.loads(annotations_file.read_text())
        if annotations_file.exists()
        else dict(points=[], human_verified=False)
    )
    for view in views.values():
        for image in view["images"].values():
            if digest(views_dir / image["path"]) != image["sha256"]:
                raise ValueError("Evaluation image hash changed")
    out = fresh_directory(root / config.get("review_directory", "review"))
    shapes = {
        "baseline": (
            baseline["camera_vertices"],
            baseline["camera_joints"],
            baseline["accepted"],
        ),
        "hawor": (
            previous["camera_vertices"],
            previous["camera_joints"],
            previous["geometry_supported"],
        ),
        "point_refined": (
            candidate["diagnostic_camera_vertices"],
            candidate["diagnostic_camera_joints"],
            diagnostic_support(candidate, baseline),
        ),
        "tracked_rigid_prior": (
            candidate["prior_camera_vertices"],
            candidate["prior_camera_joints"],
            np.isfinite(candidate["prior_camera_joints"]).all((2, 3)),
        ),
    }
    manual = {
        name: annotation_errors(annotations, j, source.frame_ids, views)
        for name, (_, j, _) in shapes.items()
    }
    heldout = dict(np.load(root / "evaluation_inputs/automatic_landmarks.npz"))
    automatic = []
    for t, frame in enumerate(heldout["frame_ids"]):
        i = int(np.flatnonzero(source.frame_ids == frame)[0])
        for h in range(2):
            if not heldout["accepted"][t, h]:
                continue
            measured = heldout["keypoints_px"][t, h]
            visible = visible_landmarks(measured, source.mask(i, config["mask_ids"][h]))
            for name, (_, joints, _) in shapes.items():
                if not np.isfinite(joints[i, h]).all():
                    continue
                errors = np.linalg.norm(
                    project(joints[i, h], source.k) - measured, axis=-1
                )
                for joint in np.flatnonzero(visible):
                    automatic.append(
                        dict(
                            variant=name,
                            frame_id=int(frame),
                            hand=h,
                            joint=int(joint),
                            error_px=float(errors[joint]),
                            refinement_attempted=bool(
                                candidate.get(
                                    "sequence_selected",
                                    candidate["refinement_attempted"],
                                )[i, h]
                            ),
                        )
                    )
    # Pair comparisons on identical available landmarks, not different coverage.
    paired = {}
    for a, b in [("hawor", "point_refined"), ("tracked_rigid_prior", "point_refined")]:
        for view in views:
            for h in range(2):
                left = {
                    (r["frame_id"], r["joint"]): r["error_px"]
                    for r in manual[a]
                    if r["hand"] == h
                    and r["view"] == view
                    and r["error_px"] is not None
                }
                right = {
                    (r["frame_id"], r["joint"]): r["error_px"]
                    for r in manual[b]
                    if r["hand"] == h
                    and r["view"] == view
                    and r["error_px"] is not None
                }
                keys = sorted(set(left) & set(right))
                paired[f"{a}_vs_{b}:{view}:{h}"] = dict(
                    frame_joint_pairs=keys,
                    before_px=distribution([left[k] for k in keys]),
                    after_px=distribution([right[k] for k in keys]),
                )
    temporal = {}
    for name, joints, supported in [
        ("baseline", baseline["world_joints"], baseline["accepted"]),
        (
            "combined_candidate",
            candidate["world_joints"],
            candidate["geometry_supported"],
        ),
    ]:
        velocity, acceleration, edges, triples = temporal_measurements(
            joints, supported, source.times
        )
        temporal[name] = [
            dict(
                wrist_speed_m_s=distribution(
                    np.linalg.norm(velocity[:, h, 0], axis=-1)[edges[:, h]]
                ),
                joint_acceleration_m_s2=distribution(
                    np.linalg.norm(acceleration[:, h], axis=-1)[triples[:, h]].ravel()
                ),
            )
            for h in range(2)
        ]
    boundaries = boundary_diagnostics(
        candidate["world_joints"],
        baseline["accepted"],
        candidate["geometry_supported"],
        source.frame_ids,
        source.times,
    )
    summary = dict(
        schema="real2sim2real.point-hand-comparison/1",
        frame_ids=source.frame_ids.tolist(),
        baseline_supported_per_hand=baseline["accepted"].sum(0).tolist(),
        candidate_supported_per_hand=candidate["geometry_supported"].sum(0).tolist(),
        recovered_per_hand=quality["recovered_per_hand"],
        refined_per_hand=quality["refined_per_hand"],
        manual_annotations_human_verified=annotations.get("human_verified", False),
        manual_annotation_reviewer=annotations.get("reviewer"),
        manual_errors=manual,
        paired_manual_comparison=paired,
        heldout_automatic_errors=automatic,
        temporal=temporal,
        temporal_coverage_differs=True,
        sequence_components=quality.get("components", []),
        sequence_admitted_per_hand=quality.get("sequence_admitted_per_hand"),
        accepted_baseline_changes_per_hand=quality.get(
            "accepted_baseline_changes_per_hand", [0, 0]
        ),
        proposed_baseline_recovery_boundaries=boundary_diagnostics(
            candidate.get("proposed_world_joints", candidate["world_joints"]),
            baseline["accepted"],
            np.isfinite(
                candidate.get("proposed_world_joints", candidate["world_joints"])
            ).all((2, 3)),
            source.frame_ids,
            source.times,
        ),
        baseline_recovery_boundaries=boundaries,
        promotion_requires=[
            "review_baseline_recovery_finger_transitions",
            "review_heldout_joint_error_tails",
            "verify_sparse_visual_annotations_and_expand_joint_coverage",
        ],
        visible_tracked_points_per_frame=(observations["weights"] > 0).sum(-1).tolist(),
        fit_track_agreement_is_not_independent_evaluation=True,
        detector_check_may_overlap_baseline_initialization=bool(
            quality.get("sequence_fitting")
        ),
        independent_3d_ground_truth_available=False,
        baseline_unchanged=True,
        promoted=False,
        control_ready=False,
        hypothesis_sha256=digest(candidate_dir / "hypotheses.npz"),
        annotations_sha256=digest(annotations_file)
        if annotations_file.exists()
        else None,
    )
    write_json(out / "comparison.json", summary)
    panels = []
    for i, frame in enumerate(source.frame_ids):
        frame_views = [("target", source.rgb(i), source.k, np.eye(4))]
        for name, view in views.items():
            image = view["images"].get(str(frame))
            if name != "target" and image:
                frame_views.append(
                    (
                        name,
                        np.asarray(
                            Image.open(views_dir / image["path"]).convert("RGB")
                        ),
                        np.asarray(view["intrinsics"]),
                        np.asarray(view["view_from_target"]),
                    )
                )
        for view_name, rgb, k, transform in frame_views:
            images = []
            labels = [
                p
                for p in annotations.get("points", [])
                if p["view"] == view_name and p["frame_id"] == frame
            ]
            for name in ("baseline", "hawor", "point_refined"):
                vertices, joints, supported = shapes[name]
                image = draw_projection(
                    rgb,
                    vertices[i],
                    joints[i],
                    candidate["faces"],
                    k,
                    transform,
                    supported[i],
                    labels=labels,
                )
                if name == "point_refined" and view_name == "target":
                    for h in range(2):
                        for point in observations["keypoints_px"][i, h]:
                            if np.isfinite(point).all():
                                cv2.circle(
                                    image,
                                    tuple(np.rint(point).astype(int)),
                                    4,
                                    (0, 255, 0),
                                    1,
                                )
                cv2.putText(
                    image,
                    name,
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                images.append(image)
            filename = f"{view_name}_{int(frame):06d}.jpg"
            Image.fromarray(np.concatenate(images, axis=1)).save(
                out / filename, quality=87
            )
            panels.append(
                f'<details><summary>Frame {frame} · {html.escape(view_name)}</summary><a href="{filename}"><img loading="lazy" src="{filename}"></a></details>'
            )
    page_summary = {
        key: summary[key]
        for key in (
            "baseline_supported_per_hand",
            "candidate_supported_per_hand",
            "recovered_per_hand",
            "refined_per_hand",
            "manual_annotations_human_verified",
            "baseline_unchanged",
            "promoted",
            "control_ready",
            "sequence_admitted_per_hand",
            "accepted_baseline_changes_per_hand",
            "sequence_components",
        )
    }
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Point-guided hand refinement</title>'
        "<style>body{font:16px system-ui;margin:25px;background:#17202b;color:#eee}img{width:100%}summary{padding:12px;cursor:pointer}a{color:#8cf}pre{white-space:pre-wrap}</style>"
        "<h1>Point-guided hand refinement</h1><p>Baseline · original HaWoR · point-guided diagnostic. Green circles: retained image tracks. Red mesh: rejected hypothesis. Magenta crosses: evaluation annotations. Images include rejected hypotheses, not a control trajectory.</p>"
        "<p>The source baseline remains unchanged. Proposed fits shown here may be rejected as a complete interval; the saved result then retains its parent. Changes to previously accepted poses are counted below. Track agreement is fitting evidence; sparse visual labels remain assistant-reviewed unless explicitly human-verified.</p>"
        "<p>Promotion is withheld. Inspect finger transitions and held-out joint-error tails; geometry support does not establish continuous accurate motion.</p>"
        '<p><a href="comparison.json">Metrics and paired errors</a></p><pre>'
        + html.escape(json.dumps(page_summary, indent=2))
        + "</pre>"
        + "".join(panels)
    )
