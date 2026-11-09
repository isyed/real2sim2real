"""Locate missing/rejected hand intervals and joint-specific image disagreement."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, load, write_json
from .metric_hands import project

JOINT_NAMES = ["wrist"] + [
    f"{finger}_{joint}"
    for finger in ("thumb", "index", "middle", "ring", "little")
    for joint in ("base", "proximal", "distal", "tip")
]


def intervals(selected, frame_ids, times):
    selected, frame_ids, times = map(np.asarray, (selected, frame_ids, times))
    if selected.shape != frame_ids.shape or times.shape != frame_ids.shape:
        raise ValueError("Matching selection and timeline required")
    ids = np.flatnonzero(selected)
    groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return [
        dict(
            first_frame=int(frame_ids[g[0]]),
            last_frame=int(frame_ids[g[-1]]),
            frames=len(g),
            start_s=float(times[g[0]]),
            end_s=float(times[g[-1]]),
        )
        for g in groups
        if len(g)
    ]


def audit(parent, output):
    parent, output = Path(parent), Path(output)
    load(parent, kinds={"metric-interaction-reconstruction"})
    if output.exists():
        raise FileExistsError("Preserve existing hand audit")
    hands = np.load(parent / "hands_metric/hands.npz")
    observations = np.load(parent / "hands2d/landmarks.npz")
    quality = json.loads((parent / "hands_metric/quality.json").read_text())
    scene = np.load(parent / "support/scene.npz")
    if not np.array_equal(hands["frame_ids"], observations["frame_ids"]):
        raise ValueError("Hand audit timelines differ")
    errors = np.linalg.norm(
        project(hands["proposed_camera_joints"], scene["intrinsics"])
        - observations["keypoints_px"],
        axis=-1,
    )
    result = {}
    for h, name in enumerate(("left", "right")):
        rejected = observations["accepted"][:, h] & ~hands["accepted"][:, h]
        finite = np.isfinite(errors[:, h]).all(1)
        reasons = Counter(
            reason
            for row in quality["records"]
            if not row[h]["accepted"]
            for reason in row[h].get("rejected_for", [row[h].get("reason", "missing")])
        )
        result[name] = dict(
            accepted_frames=int(hands["accepted"][:, h].sum()),
            rejected_intervals=intervals(
                rejected, hands["frame_ids"], hands["timestamps_s"]
            ),
            missing_intervals=intervals(
                ~observations["accepted"][:, h],
                hands["frame_ids"],
                hands["timestamps_s"],
            ),
            rejection_reasons=dict(reasons),
            rejected_joint_median_px=dict(
                zip(
                    JOINT_NAMES,
                    np.median(errors[rejected & finite, h], axis=0).tolist()
                    if (rejected & finite).any()
                    else [None] * 21,
                )
            ),
        )
    report = dict(
        schema="real2sim2real.hand-audit/1",
        total_frames=len(errors),
        hands=result,
        note="Image agreement with detected landmarks, not ground-truth joint accuracy.",
        parent_artifact_sha256=digest(parent / "artifact.json"),
    )
    write_json(output, report)
    return report
