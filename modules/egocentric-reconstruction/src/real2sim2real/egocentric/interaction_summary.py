"""Reusable coverage and consistency summaries, without hand ground truth."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction


def statistics(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0, median=None, p90=None, maximum=None)
    return dict(
        count=len(a),
        median=float(np.median(a)),
        p90=float(np.quantile(a, 0.9)),
        maximum=float(a.max()),
    )


def summarize_hands(observed, inferred, accepted, quality, contacts, tolerance=0.003):
    observed, inferred, accepted = [
        np.asarray(a, dtype=bool) for a in (observed, inferred, accepted)
    ]
    if (
        observed.ndim != 2
        or observed.shape[1] != 2
        or not observed.shape == inferred.shape == accepted.shape
    ):
        raise ValueError("Expected matching Nx2 hand validity arrays")
    if np.any(accepted & ~inferred) or np.any(inferred & ~observed):
        raise ValueError("Accepted hands require their upstream observations")
    count = len(observed)
    if len(quality) != count or len(contacts) != count:
        raise ValueError("Summary must retain the full frame denominator")
    result = {}
    for h, name in enumerate(("left", "right")):
        rows = [frame[h] for frame in quality]
        if not np.array_equal([r["accepted"] for r in rows], accepted[:, h]):
            raise ValueError("Quality rows and saved hand validity disagree")
        valid = [r for r in rows if r["accepted"]]
        reasons = Counter(
            reason
            for r in rows
            if not r["accepted"]
            for reason in r.get("rejected_for", [r.get("reason", "unavailable")])
        )
        hand_contacts = [
            frame["hands"][h] for frame in contacts if frame["hands"][h]["accepted"]
        ]
        if not np.array_equal(
            [frame["hands"][h]["accepted"] for frame in contacts], accepted[:, h]
        ):
            raise ValueError("Contact records and hand validity disagree")
        metrics = {}
        for key in (
            "reprojection_median_px",
            "reprojection_p90_px",
            "validation_surface_median_m",
            "validation_surface_p90_m",
        ):
            metrics[key] = dict(
                accepted_only=statistics([r.get(key) for r in valid]),
                all_scored_candidates=statistics([r.get(key) for r in rows]),
            )
        object_penetration = [r["object"]["penetration_depth_m"] for r in hand_contacts]
        table_penetration = [r["table"]["penetration_depth_m"] for r in hand_contacts]
        result[name] = dict(
            total_frames=count,
            detected_2d_frames=int(observed[:, h].sum()),
            inferred_3d_frames=int(inferred[:, h].sum()),
            accepted_metric_frames=int(accepted[:, h].sum()),
            metric_coverage=float(accepted[:, h].mean()) if count else 0.0,
            missing_2d_frames=int((~observed[:, h]).sum()),
            missing_model_frames_after_detection=int(
                (observed[:, h] & ~inferred[:, h]).sum()
            ),
            rejected_metric_frames=int((inferred[:, h] & ~accepted[:, h]).sum()),
            rejected_reasons=dict(reasons),
            fit_metrics=metrics,
            contact_candidate_frames=sum(
                r["object"]["contact_candidate"] for r in hand_contacts
            ),
            closest_object_distance_m=statistics(
                [r["object"]["minimum_distance_m"] for r in hand_contacts]
            ),
            object_penetration_m=statistics(object_penetration),
            table_penetration_m=statistics(table_penetration),
            object_penetration_frames_over_tolerance=sum(
                v is not None and v > tolerance for v in object_penetration
            ),
            table_penetration_frames_over_tolerance=sum(
                v is not None and v > tolerance for v in table_penetration
            ),
        )
    return result


def summarize(config):
    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    root = Path(config["output"])
    output = root / "quality_summary.json"
    if output.exists():
        raise FileExistsError("Preserve existing summary; select a new hypothesis")
    observation = np.load(root / "hands2d/landmarks.npz")
    raw = np.load(root / "hands_raw/predictions.npz")
    hands = np.load(root / "hands_metric/hands.npz")
    for value in (observation, raw, hands):
        if not np.array_equal(value["frame_ids"], source.frame_ids) or not np.allclose(
            value["timestamps_s"], source.times, rtol=0, atol=1e-8
        ):
            raise ValueError("Summary inputs have different timelines")
    quality = json.loads((root / "hands_metric/quality.json").read_text())
    contacts = json.loads((root / "contacts/diagnostics.json").read_text())
    if not np.array_equal([r["frame"] for r in contacts["records"]], source.frame_ids):
        raise ValueError("Contact frame IDs differ from the scene")
    if contacts["summary"]["sources"]["hands"] != digest(
        root / "hands_metric/hands.npz"
    ):
        raise ValueError("Contact diagnostics describe different hands")
    summary = dict(
        schema="real2sim2real.interaction-summary/1",
        total_frames=source.count,
        complete_hands=bool(hands["accepted"].all()),
        control_ready=False,
        hand_ground_truth_used=False,
        metric_kind="image_and_depth_consistency",
        metric_note="Fit statistics summarize frame-level residuals; accepted-only statistics exclude failures",
        scale_per_hand=quality["scale_per_hand"],
        contact_distance_m=contacts["summary"]["contact_distance_m"],
        penetration_tolerance_m=contacts["summary"]["penetration_tolerance_m"],
        hands=summarize_hands(
            observation["accepted"],
            raw["accepted"],
            hands["accepted"],
            quality["records"],
            contacts["records"],
            contacts["summary"]["penetration_tolerance_m"],
        ),
        object_table_max_penetration_m=contacts["summary"][
            "object_table_max_penetration_m"
        ],
        sources={
            name: digest(root / name)
            for name in (
                "hands_metric/quality.json",
                "hands_metric/hands.npz",
                "contacts/diagnostics.json",
            )
        },
    )
    write_json(output, summary)
    return summary
