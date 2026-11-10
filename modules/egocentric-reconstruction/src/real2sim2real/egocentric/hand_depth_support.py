"""Observed wrist boundaries for an explicit hand-only depth support domain.

No predicted pose, depth value or evaluation label enters boundary preparation.
Missing/withheld observations retain the original support, never an invented
wrist. The boundary is an image proxy and requires a separate image review.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import fresh_directory


def prepare_boundaries(landmarks, accepted, frame_ids, blocked_frame_ids=()):
    landmarks, accepted, frame_ids = map(np.asarray, (landmarks, accepted, frame_ids))
    if landmarks.shape != (len(frame_ids), 2, 21, 2) or accepted.shape != (
        len(frame_ids),
        2,
    ):
        raise ValueError("Matching Nx2 hand detections required")
    if not set(blocked_frame_ids).issubset(set(frame_ids.tolist())):
        raise ValueError("Withheld frame IDs must belong to the input")
    wrist = np.full((len(frame_ids), 2, 2), np.nan)
    direction = np.full_like(wrist, np.nan)
    available = np.zeros_like(accepted, bool)
    for i, frame in enumerate(frame_ids):
        if frame in blocked_frame_ids:
            continue
        for h in range(2):
            selected = landmarks[i, h, [0, 5, 9, 13, 17]]
            if not accepted[i, h] or not np.isfinite(selected).all():
                continue
            delta = selected[1:].mean(0) - selected[0]
            length = np.linalg.norm(delta)
            if not 20 <= length <= 300:
                continue
            wrist[i, h], direction[i, h] = selected[0], delta / length
            available[i, h] = True
    return dict(wrist_px=wrist, palm_direction=direction, available=available)


def signed_support_distance(points_px, wrist_px, palm_direction, margin_px):
    if (
        isinstance(margin_px, bool)
        or not np.isfinite(margin_px)
        or not 0 <= margin_px <= 100
    ):
        raise ValueError("Wrist margin must be finite in [0,100] pixels")
    points, wrist, direction = map(np.asarray, (points_px, wrist_px, palm_direction))
    if (
        points.shape[-1:] != (2,)
        or wrist.shape != (2,)
        or direction.shape != (2,)
        or not np.isfinite(points).all()
        or not np.isfinite(wrist).all()
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1, atol=1e-6)
    ):
        raise ValueError("Finite image coordinates and a unit palm direction required")
    return (points - wrist) @ direction + margin_px


def support_region(shape, boundary, i, hand, margin_px):
    if not boundary["available"][i, hand]:
        return np.ones(shape, bool)
    y, x = np.indices(shape)
    return (
        signed_support_distance(
            np.stack([x, y], -1),
            boundary["wrist_px"][i, hand],
            boundary["palm_direction"][i, hand],
            margin_px,
        )
        >= 0
    )


def load_boundaries(folder, frame_ids, times, blocked_frame_ids):
    folder = Path(folder)
    receipt = json.loads((folder / "receipt.json").read_text())
    if (
        receipt.get("schema") != "real2sim2real.hand-depth-support/1"
        or digest(folder / "boundaries.npz") != receipt["boundaries_sha256"]
    ):
        raise ValueError("Invalid hand support receipt")
    if set(receipt["blocked_frame_ids"]) != set(blocked_frame_ids):
        raise ValueError("Hand support must exclude declared held-out detector frames")
    data = dict(np.load(folder / "boundaries.npz"))
    if not np.array_equal(data["frame_ids"], frame_ids) or not np.allclose(
        data["timestamps_s"], times, atol=1e-8, rtol=0
    ):
        raise ValueError("Hand support must share the capture timeline")
    if data["available"][np.isin(frame_ids, blocked_frame_ids)].any():
        raise ValueError("Withheld wrist observations must remain unavailable")
    return data, receipt


def prepare(config):
    data = dict(np.load(config["landmarks"]))
    boundaries = prepare_boundaries(
        data["keypoints_px"],
        data["accepted"],
        data["frame_ids"],
        config["blocked_frame_ids"],
    )
    out = fresh_directory(config["output"])
    np.savez_compressed(
        out / "boundaries.npz",
        **boundaries,
        frame_ids=data["frame_ids"],
        timestamps_s=data["timestamps_s"],
    )
    write_json(
        out / "receipt.json",
        dict(
            schema="real2sim2real.hand-depth-support/1",
            boundaries_sha256=digest(out / "boundaries.npz"),
            source_landmarks_sha256=digest(config["landmarks"]),
            blocked_frame_ids=config["blocked_frame_ids"],
            available_per_hand=boundaries["available"].sum(0).tolist(),
            poses_used=False,
            depth_used=False,
            manual_labels_used=False,
            missing_behavior="retain_original_support",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
