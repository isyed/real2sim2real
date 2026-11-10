"""Optional pinned BootsTAPIR worker; no 3D poses or evaluation labels loaded."""

from pathlib import Path
import json

import cv2
import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import fresh_directory
from .hand_point_tracks import confidence, mask_support, aggregate_tracks


def run(request, check=False):
    import torch
    from tapnet.torch.tapir_model import TAPIR

    model = TAPIR(pyramid_level=1)
    weight = Path(request["weights"]["tapir"]) / "bootstapir_checkpoint_v2.pt"
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    if check:
        return
    inputs = Path(request["point_inputs"])
    data = dict(np.load(inputs / "seeds.npz", allow_pickle=False))
    frames = np.load(inputs / "rgb.npy", allow_pickle=False)
    masks = np.load(inputs / "masks.npy", allow_pickle=False)
    metadata = json.loads((inputs / "protocol.json").read_text())
    if digest(inputs / "seeds.npz") != metadata["seeds_sha256"]:
        raise ValueError("Seed provenance changed")
    for filename, key in [("rgb.npy", "rgb_sha256"), ("masks.npy", "masks_sha256")]:
        if digest(inputs / filename) != metadata[key]:
            raise ValueError("Tracking image or mask provenance changed")
    out = fresh_directory(Path(request["output"]) / "point_tracks")
    torch.manual_seed(0)
    model = model.cuda().eval()
    total, count = len(data["hand"]), len(frames)
    xy = np.full((total, count, 2), np.nan, np.float32)
    backward = xy.copy()
    scores = np.zeros((total, count), np.float32)
    reverse_scores = scores.copy()
    endpoints = np.full(total, -1, np.int32)
    crops = []
    size = int(request.get("tracking_size", 512))
    if size not in (256, 512):
        raise ValueError("Tracking resolution must be 256 or 512")

    def predict(video, queries):
        with torch.inference_mode():
            value = model(
                video=video,
                query_points=torch.as_tensor(
                    queries[None], dtype=torch.float32, device="cuda"
                ),
                query_chunk_size=32,
            )
        return (
            value["tracks"][0].cpu().numpy(),
            confidence(
                value["occlusion"][0].cpu().numpy(),
                value["expected_dist"][0].cpu().numpy(),
            ),
        )

    for h in range(2):
        ids = np.flatnonzero(data["hand"] == h)
        if not len(ids):
            continue
        _, ys, xs = np.where(masks[:, h])
        margin = 32
        x0, x1 = (
            max(0, int(xs.min()) - margin),
            min(frames.shape[2], int(xs.max()) + margin + 1),
        )
        y0, y1 = (
            max(0, int(ys.min()) - margin),
            min(frames.shape[1], int(ys.max()) + margin + 1),
        )
        # One fixed crop for the entire trajectory avoids camera-like crop motion.
        crops.append(dict(hand=h, box_xyxy=[x0, y0, x1, y1], size=size))
        factors = np.array([size / (x1 - x0), size / (y1 - y0)])
        video = np.stack([cv2.resize(im[y0:y1, x0:x1], (size, size)) for im in frames])
        video = (
            torch.as_tensor(video[None], dtype=torch.float32, device="cuda") / 127.5 - 1
        )
        queries = data["query_tyx"][ids].copy()
        queries[:, 1:] = (queries[:, 1:] - [y0, x0]) * factors[::-1]
        tracks, score = predict(video, queries)
        xy[ids], scores[ids] = tracks / factors + [x0, y0], score
        visible = mask_support(xy[ids], masks, data["hand"][ids]) & (score >= 0.8)
        reverse_queries, use = [], []
        for local, q in enumerate(ids):
            available = np.flatnonzero(visible[local])
            start = int(data["query_tyx"][q, 0])
            available = available[np.abs(available - start) >= 4]
            if not len(available):
                continue
            end = int(available[np.argmax(np.abs(available - start))])
            endpoints[q] = end
            reverse_queries.append([count - 1 - end, *tracks[local, end, ::-1]])
            use.append(q)
        if use:
            reverse, reliable = predict(
                video.flip(1), np.asarray(reverse_queries, np.float32)
            )
            backward[use] = reverse[:, ::-1] / factors + [x0, y0]
            reverse_scores[use] = reliable[:, ::-1]
        del video
        torch.cuda.empty_cache()
        print(
            f"Tracked hand {h}: {len(ids)} seeds, {len(use)} reverse queries",
            flush=True,
        )
    observations = aggregate_tracks(
        xy,
        backward,
        scores,
        reverse_scores,
        data,
        masks,
        **request.get("point_limits", {}),
    )
    np.savez_compressed(
        out / "observations.npz",
        **observations,
        frame_ids=data["frame_ids"],
        timestamps_s=data["timestamps_s"],
    )
    np.savez_compressed(
        out / "raw.npz",
        tracks_px=xy,
        reverse_tracks_px=backward,
        confidence=scores,
        reverse_confidence=reverse_scores,
        reverse_query_indices=endpoints,
        **data,
    )
    write_json(
        out / "provenance.json",
        dict(
            schema="real2sim2real.hand-point-tracks/1",
            units="target_image_pixels",
            checkpoint_sha256=digest(weight),
            seeds_sha256=digest(inputs / "seeds.npz"),
            protocol_sha256=digest(inputs / "protocol.json"),
            crops=crops,
            visible_points_per_frame=(observations["weights"] > 0).sum(-1).tolist(),
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            model="BootsTAPIR",
            independent_ground_truth=False,
            anatomical_identity_source="automatic_2d_seed_detector",
            observations_are_surface_appearance_tracks=True,
            hand_model_predictions_used=False,
            evaluation_labels_used=False,
        ),
    )
