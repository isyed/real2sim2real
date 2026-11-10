"""Pinned HaWoR temporal estimator with shared crops and calibrated camera axes.

Uses only the camera-space network. SLAM, metric-scale prediction, motion
infilling and the demo's image-centre/static-camera assumptions are bypassed.
"""

import hashlib
import inspect
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .hamer_worker import legacy_mano_compatibility, load_fixed_shape
from .hand_tracks import load_tracks
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import project


def mirrored_camera(center, principal_point, width, hand):
    center, principal_point = (
        np.array(center, dtype=float),
        np.array(principal_point, dtype=float),
    )
    if hand == 0:
        center[..., 0] = width - 1 - center[..., 0]
        principal_point[..., 0] = width - 1 - principal_point[..., 0]
    elif hand != 1:
        raise ValueError("Physical hand index must be left=0 or right=1")
    return center, principal_point


def consecutive_windows(valid, size=16):
    """Never bridge a missing crop; repeat boundary images only for model padding."""
    indices = np.flatnonzero(valid)
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        for start in range(0, len(segment), size):
            actual = segment[start : start + size]
            yield actual, np.pad(actual, (0, size - len(actual)), mode="edge")


def load_model(weights, mano_weights):
    import torch

    legacy_mano_compatibility()
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR

    root = Path(weights)
    checkpoint = root / "hawor.ckpt"
    cfg = get_config(str(root / "model_config.yaml"))
    cfg.defrost()
    cfg.MANO.MODEL_PATH = str(Path(mano_weights) / "_DATA/data/mano")
    cfg.MANO.MEAN_PARAMS = str(
        Path(inspect.getfile(HAWOR)).parents[2] / "_DATA/data/mano_mean_params.npz"
    )
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.MODEL.pop("LOAD_WEIGHTS", None)
    cfg.freeze()
    model = HAWOR(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"HaWoR checkpoint mismatch: {incompatible}")
    return model.eval(), cfg, checkpoint


def run(request, check=False):
    import cv2
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    model, cfg, checkpoint = load_model(
        request["weights"]["hawor"], request["weights"]["hamer"]
    )
    if check:
        print("HaWoR checkpoint and MANO loaded on CPU", flush=True)
        return
    source = Reconstruction(
        request["reconstruction"],
        request.get("frame_count"),
        start=request.get("frame_start", 0),
    )
    tracks = load_tracks(request["hand_tracks"], source)
    out = fresh_directory(Path(request["output"]) / "hands_raw")
    fixed = load_fixed_shape(request["fixed_shape_from"])
    n = source.count
    vertices = np.full((n, 2, 778, 3), np.nan, np.float32)
    joints = np.full((n, 2, 21, 3), np.nan, np.float32)
    uv = np.full((n, 2, 21, 2), np.nan, np.float32)
    pose = np.full((n, 2, 15, 3, 3), np.nan, np.float32)
    orient = np.full((n, 2, 1, 3, 3), np.nan, np.float32)
    betas = np.full((n, 2, 10), np.nan, np.float32)
    crop_hashes = np.full((n, 2), "", dtype="U64")
    valid = tracks["crop_valid"].copy()
    model = model.cuda()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    for h in range(2):
        cache = {}
        for i in np.flatnonzero(valid[:, h]):
            rgb = source.rgb(i)
            item = ViTDetDataset(
                cfg,
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                tracks["boxes_xyxy"][i, h : h + 1],
                np.array([h], np.float32),
                rescale_factor=request.get("hand_crop_factor", 2.0),
            )[0]
            crop_hashes[i, h] = hashlib.sha256(item["img"].tobytes()).hexdigest()
            center, principal = mirrored_camera(
                item["box_center"], source.k[:2, 2], rgb.shape[1], h
            )
            cache[i] = dict(
                img=item["img"],
                center=center.astype(np.float32),
                scale=np.float32(item["box_size"] / 200),
                img_focal=np.float32(np.sqrt(source.k[0, 0] * source.k[1, 1])),
                img_center=principal.astype(np.float32),
            )
        for actual, padded in consecutive_windows(valid[:, h]):
            batch = {
                key: torch.as_tensor(
                    np.stack([cache[i][key] for i in padded]), device="cuda"
                )[None]
                for key in cache[padded[0]]
            }
            with torch.inference_mode():
                prediction = model.forward_step(batch)["out"]
                matrices = prediction["pred_rotmat"][: len(actual)]
                shape = torch.tensor(
                    np.repeat(fixed[h : h + 1], len(actual), axis=0), device="cuda"
                )
                mano = model.mano(
                    betas=shape,
                    hand_pose=matrices[:, 1:],
                    global_orient=matrices[:, :1],
                    pose2rot=False,
                )
            v, j = mano.vertices.cpu().numpy(), mano.joints.cpu().numpy()
            translation = prediction["trans_full"][: len(actual)].cpu().numpy()
            flip = np.array([2 * h - 1, 1, 1])
            uv[actual, h] = project((j + translation) * flip, source.k)
            vertices[actual, h] = (v - j[:, :1]) * flip
            joints[actual, h] = (j - j[:, :1]) * flip
            pose[actual, h] = matrices[:, 1:].cpu().numpy()
            orient[actual, h] = matrices[:, :1].cpu().numpy()
            betas[actual, h] = prediction["pred_shape"][: len(actual)].cpu().numpy()
            print(
                f"HaWoR hand {h}, frame IDs {source.frame_ids[actual].tolist()}",
                flush=True,
            )
    faces = model.mano.faces.astype(np.int32)
    np.savez_compressed(
        out / "predictions.npz",
        vertices=vertices,
        joints=joints,
        predicted_keypoints_px=uv,
        betas=betas,
        hand_pose=pose,
        global_orient=orient,
        fixed_betas=fixed,
        accepted=valid,
        faces=np.stack([faces[:, ::-1], faces]),
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
        crop_sha256=crop_hashes,
        pose_infilled=np.zeros_like(valid),
    )
    write_json(
        out / "shape.json", dict(betas=fixed.tolist(), shared_over_sequence=True)
    )
    write_json(
        out / "provenance.json",
        dict(
            backend="HaWoR",
            upstream_commit="2c3fa0c06e54a1b192c1ff692eaa89d526cd6317",
            checkpoint_sha256=digest(checkpoint),
            mano_sha256=digest(
                Path(request["weights"]["hamer"]) / "_DATA/data/mano/MANO_RIGHT.pkl"
            ),
            geometry_shape_source="fixed_betas",
            betas_field_contains="network_shape_predictions_not_used_for_geometry",
            hand_tracks_sha256=digest(request["hand_tracks"]),
            fixed_shape_sha256=digest(request["fixed_shape_from"]),
            hand_order=["left", "right"],
            coordinate_frame="wrist_relative_camera_axes",
            image_size=source.timeline["image_size"],
            intrinsics=source.k.tolist(),
            network_focal="geometric_mean_fx_fy",
            preprocessing="shared_HaMeR_ViTDetDataset",
            block_size=16,
            boundary_padding="repeat_last_image",
            mirror_convention="reflect_camera_x_and_principal_point",
            metric_translation_estimated=False,
            infiller_used=False,
            camera_slam_used=False,
            raw_accepted_means="image_inference_available_not_metric_validated",
            accepted_per_hand=valid.sum(0).tolist(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            ground_truth_used=False,
        ),
    )
