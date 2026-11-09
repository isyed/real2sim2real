"""Audit missing 2D hands using fresh detections in rotated, neighbor-guided crops.

Neighbors supply only a search box. Every reported landmark is a new image
detection; no landmark trajectory is interpolated and no 3D hand is invented.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np

from real2sim2real.artifacts import digest, load, write_json
from .hand_landmarks import mask_overlap, draw_hand
from .interaction_io import Reconstruction, fresh_directory


def restore_rotation(points, width, height, turns):
    x, y = np.asarray(points).T
    if turns == 0:
        return np.stack((x, y), -1)
    if turns == 1:
        return np.stack((width - 1 - y, x), -1)
    if turns == 2:
        return np.stack((width - 1 - x, height - 1 - y), -1)
    if turns == 3:
        return np.stack((y, height - 1 - x), -1)
    raise ValueError("Rotation must be an integer number of quarter turns")


def consensus(candidates, *, median_px=10, p90_px=20):
    """Require agreeing detections from distinct crop/rotation requests."""
    best = None
    for i, row in enumerate(candidates):
        votes, residuals = [], []
        for j, other in enumerate(candidates):
            distance = np.linalg.norm(
                np.asarray(row["keypoints_px"]) - other["keypoints_px"], axis=1
            )
            if (
                np.median(distance) <= median_px
                and np.quantile(distance, 0.9) <= p90_px
            ):
                votes.append(j)
                residuals.append(float(np.median(distance)))
        views = {tuple(candidates[j]["view"]) for j in votes}
        score = (len(views), row["mask_overlap"], -sum(residuals))
        if best is None or score > best[0]:
            best = score, i, votes
    if best is None or best[0][0] < 2:
        return None
    return dict(candidate_index=best[1], votes=best[2], agreeing_views=best[0][0])


def probe(config):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = config["hand_observation_probe"]
    parent, model = Path(options["parent"]), Path(options["model"])
    load(parent, kinds={"metric-interaction-reconstruction"})
    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    observations = np.load(parent / "hands2d/landmarks.npz")
    if not np.array_equal(
        observations["frame_ids"], source.frame_ids
    ) or not np.array_equal(observations["timestamps_s"], source.times):
        raise ValueError("Observation probe requires the parent's exact timeline")
    for name, sha in json.loads((parent / "inputs.json").read_text())["files"].items():
        if digest(source.path(name)) != sha:
            raise ValueError("Probe input differs from parent")
    out = fresh_directory(Path(config["output"]) / "observation_probe")
    settings = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
    )
    found = np.zeros_like(observations["accepted"])
    keypoints = np.full_like(observations["keypoints_px"], np.nan)
    records = []
    started = time.monotonic()
    timeout = options.get("timeout_s", 300)
    if not np.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("Probe requires a finite CPU deadline of at most one hour")
    with vision.HandLandmarker.create_from_options(settings) as detector:
        for i, h in np.argwhere(~observations["accepted"]):
            if time.monotonic() - started > timeout:
                raise TimeoutError("Missing-hand observation probe exceeded deadline")
            rgb = source.rgb(i)
            height, width = rgb.shape[:2]
            masks = [
                source.mask(i, j)
                for j in config["hand_landmarks"].get("mask_ids", [2, 3])
            ]
            mask = (
                cv2.dilate(masks[h].astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            )
            other_mask = (
                cv2.dilate(masks[1 - h].astype(np.uint8), np.ones((11, 11), np.uint8))
                > 0
            )
            neighbors = np.flatnonzero(observations["accepted"][:, h])
            prior = neighbors[neighbors < i][-1:]
            future = neighbors[neighbors > i][:1]
            boxes = [
                observations["boxes_xyxy"][j, h]
                for j in np.r_[prior, future]
                if abs(source.times[j] - source.times[i]) <= 1.5
            ]
            if len(boxes) == 2:
                boxes.append(
                    np.r_[
                        np.minimum(boxes[0][:2], boxes[1][:2]),
                        np.maximum(boxes[0][2:], boxes[1][2:]),
                    ]
                )
            crops = set()
            for box in boxes:
                centre = (box[:2] + box[2:]) / 2
                side = max(box[2:] - box[:2]) * 1.6
                x0, y0 = np.maximum(0, np.floor(centre - side / 2)).astype(int)
                x1, y1 = np.minimum([width, height], np.ceil(centre + side / 2)).astype(
                    int
                )
                if x1 - x0 > 20 and y1 - y0 > 20:
                    crops.add((int(x0), int(y0), int(x1), int(y1)))
            candidates = []
            for crop_id, (x0, y0, x1, y1) in enumerate(sorted(crops)):
                for turns in range(4):
                    crop = np.ascontiguousarray(np.rot90(rgb[y0:y1, x0:x1], turns))
                    result = detector.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                    )
                    for landmarks in result.hand_landmarks:
                        uv = np.asarray(
                            [
                                [p.x * crop.shape[1], p.y * crop.shape[0]]
                                for p in landmarks
                            ]
                        )
                        uv = restore_rotation(uv, x1 - x0, y1 - y0, turns) + [x0, y0]
                        if not np.isfinite(uv).all():
                            continue
                        overlap, wrong = (
                            mask_overlap(uv, mask),
                            mask_overlap(uv, other_mask),
                        )
                        if overlap >= 0.65 and wrong <= 0.3:
                            candidates.append(
                                dict(
                                    keypoints_px=uv.tolist(),
                                    mask_overlap=overlap,
                                    other_hand_overlap=wrong,
                                    view=[crop_id, turns],
                                )
                            )
            agree = consensus(candidates)
            if agree is not None:
                found[i, h] = True
                keypoints[i, h] = candidates[agree["candidate_index"]]["keypoints_px"]
            record = dict(
                frame=int(source.frame_ids[i]),
                hand=int(h),
                found=bool(found[i, h]),
                mask_pixels=int(masks[h].sum()),
                crops=[list(c) for c in sorted(crops)],
                consensus=agree,
                candidates=candidates,
            )
            records.append(record)
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            draw_hand(overlay, keypoints[i, h], (255, 100, 255))
            cv2.imwrite(str(out / f"{i:06d}-{h}.jpg"), overlay)
            write_json(out / "attempts.json", records)
            print(
                f"probe {i}, hand {h}: {len(candidates)} candidates, consensus={agree is not None}",
                flush=True,
            )
    np.savez_compressed(
        out / "candidates.npz",
        keypoints_px=keypoints,
        accepted=found,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    report = dict(
        schema="real2sim2real.hand-observation-probe/1",
        recovered_2d_per_hand=found.sum(0).tolist(),
        missing_2d_per_hand=(~observations["accepted"]).sum(0).tolist(),
        metric_hands_recovered=False,
        inference_is_cpu=True,
        model_sha256=digest(model),
        parent_artifact_sha256=digest(parent / "artifact.json"),
        elapsed_s=time.monotonic() - started,
        note="Augmentation agreement is not independent accuracy validation. Neighbor boxes guide search only; output landmarks are fresh image detections.",
    )
    write_json(out / "summary.json", report)
    return report
