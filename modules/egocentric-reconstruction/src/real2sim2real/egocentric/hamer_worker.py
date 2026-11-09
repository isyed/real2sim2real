"""Pinned HaMeR inference with explicit crops and sequence-constant hand shape.

HaMeR is installed in the isolated worker image. Its learned camera parameters
are used only to retain image-space predictions, not as calibrated metric pose.
"""

import inspect
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory


def load_fixed_shape(path):
    """JSON numbers must match the float32 MANO model's tensor dtype."""
    betas = np.asarray(json.loads(Path(path).read_text())["betas"], dtype=np.float32)
    if betas.shape != (2, 10) or not np.isfinite(betas).all():
        raise ValueError("Expected two finite ten-coefficient hand shapes")
    return betas


def load_model(weights):
    import torch

    # MANO's original pickle contains Chumpy values; compatibility is confined to
    # this legacy loader, before importing Chumpy. No pickle from video inputs.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in dict(
        bool=bool,
        int=int,
        float=float,
        complex=complex,
        object=object,
        str=str,
        unicode=str,
    ).items():
        if name not in np.__dict__:
            setattr(np, name, value)
    from hamer.configs import get_config
    from hamer.models import HAMER

    weights = Path(weights)
    mano = weights / "_DATA/data/mano/MANO_RIGHT.pkl"
    if not mano.is_file():
        raise FileNotFoundError(f"Supply the licensed MANO_RIGHT.pkl model at {mano}")
    checkpoint = weights / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    cfg = get_config(str(checkpoint.parent.parent / "model_config.yaml"))
    cfg.defrost()
    cfg.MANO.MODEL_PATH = str(mano.parent)
    cfg.MANO.MEAN_PARAMS = str(weights / "_DATA/data/mano_mean_params.npz")
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.freeze()
    model = HAMER(cfg, init_renderer=False)
    # The official upstream checkpoint is a Lightning archive with config types.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise ValueError(f"Missing HaMeR weights: {incompatible.missing_keys}")
    return model.eval(), cfg, checkpoint, mano


def run(request, check=False):
    import torch
    import cv2
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    model, cfg, checkpoint, mano_asset = load_model(request["weights"]["hamer"])
    if check:
        print("HaMeR checkpoint and MANO loaded on CPU", flush=True)
        return
    source = Reconstruction(request["reconstruction"], request.get("frame_count"))
    root = Path(request["output"])
    observations = np.load(root / "hands2d/landmarks.npz")
    if not np.array_equal(observations["frame_ids"], source.frame_ids):
        raise ValueError("Hand observations do not match reconstruction frames")
    out = fresh_directory(root / "hands_raw")
    model = model.cuda()
    n = source.count
    betas = np.full((n, 2, 10), np.nan, np.float32)
    pose = np.full((n, 2, 15, 3, 3), np.nan, np.float32)
    orient = np.full((n, 2, 1, 3, 3), np.nan, np.float32)
    predicted_uv = np.full((n, 2, 21, 2), np.nan, np.float32)
    model_joints = np.full((n, 2, 21, 3), np.nan, np.float32)
    model_vertices = np.full((n, 2, 778, 3), np.nan, np.float32)
    accepted = observations["accepted"].copy()
    torch.manual_seed(0)
    for i in range(n):
        ids = np.flatnonzero(accepted[i])
        if not len(ids):
            continue
        dataset = ViTDetDataset(
            cfg,
            cv2.cvtColor(source.rgb(i), cv2.COLOR_RGB2BGR),
            observations["boxes_xyxy"][i, ids],
            ids.astype(np.float32),
            rescale_factor=request.get("hand_crop_factor", 2.0),
        )
        for batch in torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        ):
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.inference_mode():
                prediction = model(batch)
            for j, h in enumerate(ids):
                params = prediction["pred_mano_params"]
                betas[i, h] = params["betas"][j].cpu().numpy()
                pose[i, h] = params["hand_pose"][j].cpu().numpy()
                orient[i, h] = params["global_orient"][j].cpu().numpy()
                points = prediction["pred_keypoints_2d"][j].cpu().numpy().copy()
                points[:, 0] *= 2 * h - 1
                predicted_uv[i, h] = (
                    points * float(batch["box_size"][j])
                    + batch["box_center"][j].cpu().numpy()
                )
        if i % 20 == 0:
            print(f"HaMeR {i + 1}/{n}", flush=True)
    if not accepted.any(0).all():
        raise ValueError("Each hand needs at least one supported model observation")
    fixed_betas = np.stack(
        [np.median(betas[accepted[:, h], h], axis=0) for h in range(2)]
    )
    if request.get("fixed_shape_from"):
        fixed_betas = load_fixed_shape(request["fixed_shape_from"])
    for i, h in np.argwhere(accepted):
        with torch.inference_mode():
            output = model.mano(
                betas=torch.tensor(fixed_betas[h : h + 1], device="cuda"),
                hand_pose=torch.tensor(pose[i, h][None], device="cuda"),
                global_orient=torch.tensor(orient[i, h][None], device="cuda"),
                pose2rot=False,
            )
        vertices = output.vertices[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()
        vertices -= joints[0].copy()
        joints -= joints[0].copy()
        vertices[:, 0] *= 2 * h - 1
        joints[:, 0] *= 2 * h - 1
        model_vertices[i, h], model_joints[i, h] = vertices, joints
    faces_right = model.mano.faces.astype(np.int32)
    np.savez_compressed(
        out / "predictions.npz",
        vertices=model_vertices,
        joints=model_joints,
        predicted_keypoints_px=predicted_uv,
        betas=betas,
        hand_pose=pose,
        global_orient=orient,
        fixed_betas=fixed_betas,
        accepted=accepted,
        faces=np.stack([faces_right[:, ::-1], faces_right]),
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "shape.json", dict(betas=fixed_betas.tolist(), shared_over_sequence=True)
    )
    write_json(
        out / "provenance.json",
        dict(
            backend="HaMeR",
            upstream_commit="3a01849f4148352e9260b69bf28b65d1671a4905",
            checkpoint_sha256=digest(checkpoint),
            mano_sha256=digest(mano_asset),
            observations_sha256=digest(root / "hands2d/landmarks.npz"),
            fixed_shape_sha256=digest(request["fixed_shape_from"])
            if request.get("fixed_shape_from")
            else None,
            coordinate_frame="wrist_relative_camera_axes",
            metric_translation_estimated=False,
            hand_order=["left", "right"],
            accepted_per_hand=accepted.sum(0).tolist(),
            total_frames=n,
            ground_truth_used=False,
        ),
    )
