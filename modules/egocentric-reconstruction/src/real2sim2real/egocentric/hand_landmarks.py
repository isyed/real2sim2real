"""Optional CPU 2D hand observations with identity assigned by input masks.

MediaPipe's learned 3D landmarks are retained as a separate prior, never treated
as camera translations or metric stereo observations. Missing hands stay NaN.
"""

from pathlib import Path
import importlib.metadata

import cv2
import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory


HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def mask_overlap(points, mask):
    p = np.rint(points).astype(int)
    valid = (
        (p[:, 0] >= 0)
        & (p[:, 1] >= 0)
        & (p[:, 0] < mask.shape[1])
        & (p[:, 1] < mask.shape[0])
    )
    inside = np.zeros(len(p), bool)
    inside[valid] = mask[p[valid, 1], p[valid, 0]]
    return float(inside.mean())


def draw_hand(image, points, color):
    if not np.isfinite(points).all():
        return
    p = np.rint(points).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    for point in p:
        cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)


def detect(config):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    options = config["hand_landmarks"]
    out = fresh_directory(Path(config["output"]) / "hands2d")
    model = Path(options["model"])
    hands = options.get("mask_ids", [2, 3])  # Explicit physical left/right order.
    uv = np.full((source.count, 2, 21, 2), np.nan, np.float32)
    prior = np.full((source.count, 2, 21, 3), np.nan, np.float32)
    accepted = np.zeros((source.count, 2), bool)
    scores = np.zeros((source.count, 2), np.float32)
    boxes = np.full((source.count, 2, 4), np.nan, np.float32)
    records = []
    settings = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=options.get("detection_confidence", 0.4),
        min_hand_presence_confidence=options.get("presence_confidence", 0.4),
    )
    with vision.HandLandmarker.create_from_options(settings) as detector:
        for i in range(source.count):
            rgb = source.rgb(i)
            height, width = rgb.shape[:2]
            masks = [source.mask(i, j) for j in hands]
            candidates = []
            crops = [(0, 0, width, height)]
            for mask in masks:
                yy, xx = np.nonzero(mask)
                if len(xx) < 100:
                    continue
                pad = int(0.25 * max(np.ptp(xx), np.ptp(yy)))
                crops.append(
                    (
                        max(0, xx.min() - pad),
                        max(0, yy.min() - pad),
                        min(width, xx.max() + pad + 1),
                        min(height, yy.max() + pad + 1),
                    )
                )
            for x0, y0, x1, y1 in crops:
                crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                result = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                )
                for landmarks, world, handedness in zip(
                    result.hand_landmarks,
                    result.hand_world_landmarks,
                    result.handedness,
                ):
                    points = np.array(
                        [
                            [p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0]
                            for p in landmarks
                        ]
                    )
                    candidates.append(
                        (
                            points,
                            np.array([[p.x, p.y, p.z] for p in world]),
                            handedness[0].category_name,
                            float(handedness[0].score),
                        )
                    )
            row = []
            for h, mask in enumerate(masks):
                dilated = (
                    cv2.dilate(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                )
                ranked = sorted(
                    [
                        (mask_overlap(c[0], dilated), j)
                        for j, c in enumerate(candidates)
                    ],
                    reverse=True,
                )
                if ranked and ranked[0][0] >= options.get("minimum_mask_overlap", 0.65):
                    overlap, j = ranked[0]
                    p, world, label, label_score = candidates[j]
                    uv[i, h], prior[i, h] = p, world
                    accepted[i, h], scores[i, h] = True, overlap
                    boxes[i, h] = [*p.min(0), *p.max(0)]
                    row.append(
                        dict(
                            hand=["left", "right"][h],
                            accepted=True,
                            mask_overlap=overlap,
                            detector_handedness=label,
                            handedness_score=label_score,
                        )
                    )
                else:
                    row.append(
                        dict(
                            hand=["left", "right"][h],
                            accepted=False,
                            reason="no_mask_supported_detection",
                        )
                    )
            records.append(dict(frame=i, hands=row))
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for h, color in enumerate([(255, 160, 30), (40, 190, 255)]):
                draw_hand(overlay, uv[i, h], color)
            cv2.imwrite(str(out / f"{i:06d}.jpg"), overlay)
            if i % 25 == 0:
                print(
                    f"hands2d {i + 1}/{source.count}: {accepted[: i + 1].sum(axis=0).tolist()}",
                    flush=True,
                )
    np.savez_compressed(
        out / "landmarks.npz",
        keypoints_px=uv,
        learned_joints_prior=prior,
        boxes_xyxy=boxes,
        accepted=accepted,
        mask_overlap=scores,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "quality.json",
        dict(
            schema="real2sim2real.hand-observations/1",
            backend="mediapipe",
            version=importlib.metadata.version("mediapipe"),
            model_sha256=digest(model),
            hand_order=["left", "right"],
            mask_ids=hands,
            accepted_per_hand=accepted.sum(0).tolist(),
            total_frames=source.count,
            learned_prior_is_metric_observation=False,
            records=records,
        ),
    )
