"""Hand localisation independent of joint detection, with explicit provenance.

Boxes are pixel edge coordinates [x0, y0, x1, y1]. Physical identities come from
the supplied left/right masks. A crop is an inference input, never a 3D observation.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from real2sim2real.artifacts import write_json


def select_crops(boxes, confidence, masks, *, minimum_pixels=80, minimum_purity=0.7):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    confidence = np.asarray(confidence, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != 2 or confidence.shape != (len(boxes),):
        raise ValueError("Expected two physical hand masks and one score per box")
    height, width = masks.shape[1:]
    scores = np.zeros((2, len(boxes)))
    for j, box in enumerate(boxes):
        if not np.isfinite(box).all() or not np.isfinite(confidence[j]):
            continue
        x0, y0, x1, y1 = np.rint(box).astype(int)
        x0, x1 = np.clip([x0, x1], 0, width)
        y0, y1 = np.clip([y0, y1], 0, height)
        if x1 <= x0 or y1 <= y0:
            continue
        pixels = masks[:, y0:y1, x0:x1].sum((1, 2))
        for h in range(2):
            purity = pixels[h] / max(1, pixels.sum())
            if pixels[h] >= minimum_pixels and purity >= minimum_purity:
                scores[h, j] = purity * confidence[j]
    selected = np.full((2, 4), np.nan, np.float32)
    source = np.full(2, "missing", dtype="U16")
    selected_score = np.zeros(2, np.float32)
    if len(boxes):
        hands, candidates = linear_sum_assignment(-scores)
        for h, j in zip(hands, candidates):
            if scores[h, j] > 0:
                selected[h], source[h], selected_score[h] = (
                    boxes[j],
                    "detector",
                    scores[h, j],
                )
    for h in range(2):
        if source[h] == "missing":
            y, x = np.nonzero(masks[h])
            if len(x) >= minimum_pixels:
                selected[h] = [x.min(), y.min(), x.max() + 1, y.max() + 1]
                source[h] = "mask"
    return selected, source, selected_score


def validate_tracks(data, frame_ids, timestamps_s):
    n = len(frame_ids)
    if not np.array_equal(data["frame_ids"], frame_ids) or not np.allclose(
        data["timestamps_s"], timestamps_s, atol=1e-8, rtol=0
    ):
        raise ValueError("Hand tracks must match the exact reconstruction window")
    boxes = np.asarray(data["boxes_xyxy"])
    valid = np.asarray(data["crop_valid"])
    sources = np.asarray(data["crop_source"])
    if (
        boxes.shape != (n, 2, 4)
        or valid.shape != (n, 2)
        or valid.dtype != bool
        or sources.shape != (n, 2)
    ):
        raise ValueError("Invalid hand-track shapes or validity dtype")
    if not np.isin(sources, ["detector", "mask", "missing"]).all():
        raise ValueError("Unknown crop provenance")
    if not np.array_equal(valid, sources != "missing"):
        raise ValueError("Crop validity and provenance disagree")
    b = boxes[valid]
    if not np.isfinite(b).all() or (b[:, 2:] <= b[:, :2]).any():
        raise ValueError("Accepted crops must have finite positive extents")
    if np.isfinite(boxes[~valid]).any():
        raise ValueError("Missing crops must remain NaN")
    return data


def load_tracks(path, source):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    return validate_tracks(data, source.frame_ids, source.times)


def detect(request, check=False):
    import cv2
    import torch
    from ultralytics import YOLO

    from real2sim2real.artifacts import digest
    from .interaction_io import Reconstruction, fresh_directory

    checkpoint = Path(request["weights"]["hawor"]) / "detector.pt"
    model = YOLO(str(checkpoint))
    if check:
        print("WiLoR/HaWoR detector loaded on CPU", flush=True)
        return
    source = Reconstruction(
        request["reconstruction"],
        request.get("frame_count"),
        start=request.get("frame_start", 0),
    )
    out = fresh_directory(Path(request["output"]) / "hand_tracks")
    boxes = np.full((source.count, 2, 4), np.nan, np.float32)
    origins = np.full((source.count, 2), "missing", dtype="U16")
    confidence = np.zeros((source.count, 2), np.float32)
    records = []
    for i in range(source.count):
        with torch.inference_mode():
            result = model.predict(
                cv2.cvtColor(source.rgb(i), cv2.COLOR_RGB2BGR),
                device=0,
                conf=request.get("detector_confidence", 0.2),
                verbose=False,
            )[0]
        detected = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        masks = [source.mask(i, h) for h in request.get("mask_ids", [2, 3])]
        boxes[i], origins[i], confidence[i] = select_crops(detected, scores, masks)
        records.append(
            dict(
                frame_id=int(source.frame_ids[i]),
                boxes_xyxy=detected.tolist(),
                confidence=scores.tolist(),
                detector_classes=result.boxes.cls.cpu().numpy().tolist(),
                selected_source=origins[i].tolist(),
            )
        )
        print(f"hand crops {i + 1}/{source.count}: {origins[i].tolist()}", flush=True)
    data = dict(
        boxes_xyxy=boxes,
        crop_source=origins,
        crop_valid=origins != "missing",
        detector_score=confidence,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    validate_tracks(data, source.frame_ids, source.times)
    np.savez_compressed(out / "tracks.npz", **data)
    write_json(
        out / "provenance.json",
        dict(
            schema="real2sim2real.hand-tracks/1",
            backend="WiLoR detector used by HaWoR",
            checkpoint_sha256=digest(checkpoint),
            hand_order=["left", "right"],
            identity_source="supplied_physical_hand_masks",
            metric_observations=False,
            interpolation_used=False,
            records=records,
        ),
    )
