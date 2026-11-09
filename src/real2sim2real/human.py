"""Explicit MHR/SOMA conversion with per-frame identity preservation.
Adapted from NVIDIA Apache-2.0 exporters; model decoding constants use the SAM license.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Mapping
import numpy as np
from scipy.spatial.transform import Rotation

CV_NATIVE = np.diag([1.0, -1.0, -1.0])

INITIALIZATION_DIMS = {
    "mhr_global_rot6d": 6,
    "mhr_trans": 3,
    "mhr_body_pose_cont": 260,
    "mhr_hand": 108,
    "mhr_shape": 45,
    "mhr_scale": 28,
    "mhr_face": 72,
}

BUFFER_SHAPES = {
    "scale_mean": (68,),
    "scale_comps": (28, 68),
    "hand_pose_mean": (54,),
    "hand_pose_comps": (54, 54),
    "hand_joint_idxs_left": (27,),
    "hand_joint_idxs_right": (27,),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != tuple(shape) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite {tuple(shape)}, got {result.shape}")
    return result


def _frames(value, count, dimension, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (dimension,):
        value = value[None]
    if value.shape == (1, dimension):
        value = np.broadcast_to(value, (count, dimension)).copy()
    return _finite(value, (count, dimension), name)


def rigid_transform(value, name="transform"):
    value = _finite(value, (4, 4), name)
    if (
        not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(
            value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(value[:3, :3]), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError(
            f"{name}: expected a rigid right-handed transform, without scale"
        )
    return value


def object_pose_sequence(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not len(value):
        raise ValueError("object poses must be nonempty [T,4,4]")
    for i, matrix in enumerate(value):
        rigid_transform(matrix, f"object pose {i}")
    return value


def normalize_predicted_object_rotations(value):
    """Remove bounded float32 tracking drift; reject deformation/reflection.

    The initialization exporter already admits rotation roundoff up to 1e-4.
    Project only within that same input contract, then use the strict rigid
    checks downstream. Source predictions and every translation are preserved.
    """
    poses = np.asarray(value, dtype=np.float64).copy()
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not len(poses)
        or not np.isfinite(poses).all()
    ):
        raise ValueError("Expected finite object poses [T,4,4]")
    rotation = poses[:, :3, :3]
    defect = np.max(np.abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3)))
    determinants = np.linalg.det(rotation)
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or defect > 1e-4
        or np.max(np.abs(determinants - 1)) > 1e-4
    ):
        raise ValueError("Object rotations exceed the initialization roundoff contract")
    u, _, vh = np.linalg.svd(rotation)
    normalized = u @ vh
    delta = float(np.max(np.abs(normalized - rotation)))
    if np.any(np.linalg.det(normalized) <= 0) or delta > 1e-4:
        raise ValueError("Object rotation projection exceeds bounded roundoff")
    poses[:, :3, :3] = normalized
    object_pose_sequence(poses)
    return poses, dict(
        method="nearest_SO3_for_bounded_float32_roundoff",
        frames=len(poses),
        maximum_input_orthogonality_error=float(defect),
        maximum_rotation_element_change=delta,
        maximum_input_determinant_error=float(np.max(np.abs(determinants - 1))),
        acceptance_limit=1e-4,
        translation_changed=False,
        source_predictions_changed=False,
    )


def align_native_vertices_cm(vertices, world_transform):
    """Apply a metric CV-world transform to MHR-native centimetre vertices."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("MHR vertices must be finite [T,V,3] in native centimetres")
    transform = rigid_transform(world_transform)
    native_rotation = CV_NATIVE @ transform[:3, :3] @ CV_NATIVE
    native_translation_cm = CV_NATIVE @ transform[:3, 3] * 100.0
    return (vertices @ native_rotation.T + native_translation_cm).astype(np.float32)


def mhr_parameter_container(pose, scales, shape):
    """Pack already-decoded 136+68 arrays without changing their units."""
    pose = np.asarray(pose, dtype=np.float64)
    if (
        pose.ndim != 2
        or pose.shape[1] != 136
        or not len(pose)
        or not np.isfinite(pose).all()
    ):
        raise ValueError("Decoded MHR pose must be finite [T,136]")
    count = len(pose)
    return {
        "mhr_model_params": np.concatenate(
            (pose, _frames(scales, count, 68, "scales")), axis=-1
        ).astype(np.float32),
        "shape_params": _frames(shape, count, 45, "shape").astype(np.float32),
    }


def _global_rotation6d(value):
    # CARI4D global 6D uses interleaved matrix columns, unlike compact body/hand.
    columns = value.reshape(-1, 3, 2)
    first, second = columns[..., 0], columns[..., 1]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Degenerate global 6D rotation")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Degenerate global 6D rotation")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def decode_initialization_mhr(arrays: Mapping, buffers: Mapping):
    """Decode native stage-5 fields with pinned SAM head buffers, preserving identity.

    This is parameter decoding, not image inference or model fine-tuning. No
    frame averaging or replacement with clean human data occurs.
    """
    from .human_decode import (
        compact_cont_to_model_params_body,
        compact_cont_to_model_params_hand,
    )

    root = np.asarray(arrays["mhr_global_rot6d"])
    if root.ndim != 2 or root.shape[1] != 6 or not len(root):
        raise ValueError("mhr_global_rot6d must be nonempty [T,6]")
    count = len(root)
    data = {
        k: _frames(arrays[k], count, dim, k) for k, dim in INITIALIZATION_DIMS.items()
    }
    b = {k: _finite(buffers[k], shape, k) for k, shape in BUFFER_SHAPES.items()}
    indices = []
    for side in ("left", "right"):
        idx = b[f"hand_joint_idxs_{side}"]
        if not np.array_equal(idx, idx.astype(np.int64)) or np.any(
            (idx < 6) | (idx >= 130)
        ):
            raise ValueError("Invalid packed hand parameter indices")
        indices.append(idx.astype(np.int64))
    if len(np.unique(np.concatenate(indices))) != 54:
        raise ValueError("Hand parameter indices overlap")
    body = compact_cont_to_model_params_body(data["mhr_body_pose_cont"])
    pose = np.concatenate(
        (
            data["mhr_trans"] @ CV_NATIVE * 10.0,
            Rotation.from_matrix(_global_rotation6d(data["mhr_global_rot6d"])).as_euler(
                "ZYX"
            ),
            body[:, :130],
        ),
        axis=-1,
    )
    for hand, idx in zip(np.split(data["mhr_hand"], 2, axis=-1), indices):
        continuous = b["hand_pose_mean"] + hand @ b["hand_pose_comps"]
        pose[:, idx] = compact_cont_to_model_params_hand(continuous)
    scales = b["scale_mean"] + data["mhr_scale"] @ b["scale_comps"]
    result = mhr_parameter_container(pose, scales, data["mhr_shape"])
    result["face_params"] = data["mhr_face"].astype(np.float32)
    return result


def load_mhr_buffers(path):
    """Read only a compact local tensor state, with PyTorch's safe loader."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    result = {}
    for key, value in state.items():
        key = key.removeprefix("head_pose.")
        if key in BUFFER_SHAPES:
            if value.is_sparse:
                value = value.to_dense()
            result[key] = value.detach().cpu().numpy()
    missing = set(BUFFER_SHAPES) - set(result)
    if missing:
        raise ValueError(f"Compact MHR buffers are missing {sorted(missing)}")
    return result


def invert_mhr_to_soma(
    packed,
    *,
    mhr_model,
    soma_assets,
    output,
    world_transform,
    device="cpu",
    batch_size=16,
    body_iters=2,
    full_iters=1,
    lie_iters=3,
    inversion_low_lod=True,
):
    """Run the explicit local MHR JIT and SOMA inversion; no image/refinement stages.

    Source inversion code: NVIDIA ``export_soma.py`` (Apache-2.0). A world-frame
    transform is applied to generated vertices before fitting; body parameters
    and their flexible-bone/identity values are kept intact.
    """
    import torch
    from soma import SOMALayer
    from soma.geometry.rig_utils import remove_joint_orient_local
    from soma.geometry.transforms import matrix_to_rotvec
    from soma.io import save_soma_npz
    from soma.pose_inversion import PoseInversion
    from soma.units import Unit

    model_path, asset_root, output = (
        Path(mhr_model).resolve(),
        Path(soma_assets).resolve(),
        Path(output),
    )
    if not model_path.is_file() or not asset_root.is_dir():
        raise FileNotFoundError(
            "Explicit local MHR model and pinned SOMA assets are required"
        )
    if output.exists():
        raise FileExistsError(output)
    required = [
        "SOMA_neutral.npz",
        "correctives_model.pt",
        "MHR/mhr_model_lod1.pt",
        "MHR/base_body_lod1.obj",
        "MHR/SOMA_wrap_lod1.obj",
    ]
    if inversion_low_lod:
        required += ["MHR/mhr_model_lod6.pt", "MHR/base_body_lod6.obj"]
    for relative in required:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing/unresolved pinned SOMA asset: {path}")
        with path.open("rb") as stream:
            if stream.read(48).startswith(b"version https://git-lfs.github.com/spec/"):
                raise FileNotFoundError(f"Unresolved pinned SOMA asset: {path}")
    with np.load(asset_root / "SOMA_neutral.npz", allow_pickle=False) as archive:
        if "joint_names" not in archive.files or "bind_pose_world" not in archive.files:
            raise ValueError(
                "Pinned SOMA assets must contain the embedded 78-joint rig"
            )
    if batch_size <= 0 or any(x < 0 for x in (body_iters, full_iters, lie_iters)):
        raise ValueError("Invalid inversion batch/iteration limits")
    transform = rigid_transform(world_transform)
    params = np.asarray(packed["mhr_model_params"], dtype=np.float32)
    if (
        params.ndim != 2
        or params.shape[1] != 204
        or not len(params)
        or not np.isfinite(params).all()
    ):
        raise ValueError("Packed MHR parameters must be finite [T,204]")
    count = len(params)
    shape = _frames(packed["shape_params"], count, 45, "shape").astype(np.float32)
    face = _frames(packed.get("face_params", np.zeros(72)), count, 72, "face").astype(
        np.float32
    )
    model = torch.jit.load(str(model_path), map_location=device).eval()
    soma = SOMALayer(
        data_root=str(asset_root),
        identity_model_type="mhr",
        device=device,
        mode="warp",
        output_unit=Unit.CENTIMETERS,
        enable_procedural_transforms=False,
    )
    inverse = PoseInversion(soma, low_lod=inversion_low_lod)
    rotations, translations, errors = [], [], []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        identity = torch.from_numpy(shape[start:stop]).to(device)
        model_params = torch.from_numpy(params[start:stop]).to(device)
        with torch.no_grad():
            vertices, _ = model(
                identity, model_params, torch.from_numpy(face[start:stop]).to(device)
            )
        vertices = torch.from_numpy(
            align_native_vertices_cm(vertices.detach().cpu().numpy(), transform)
        ).to(device)
        inverse.prepare_identity(
            identity,
            model_params[:, 136:],
            kwargs={"bone_length_flexibles": model_params[:, 130:136]},
        )
        result = inverse.fit(
            vertices,
            body_iters=body_iters,
            full_iters=full_iters,
            finger_iters=0,
            lie_iters=lie_iters,
            lie_lambda=0.1,
            autograd_iters=0,
            leaf_weight=1.0,
            batch_size=None,
        )
        rotations.append(result["rotations"].detach().cpu())
        translations.append(result["root_translation"].detach().cpu())
        errors.append(result["per_vertex_error"].detach().cpu())
        print(f"SOMA conversion: {stop}/{count} frames", flush=True)
    rotations, translations = torch.cat(rotations), torch.cat(translations)
    error = torch.cat(errors).numpy()
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite SOMA inversion diagnostics")
    rel = remove_joint_orient_local(
        rotations.to(soma._t_pose_orient.device),
        soma._t_pose_orient,
        soma._t_pose_orient_parent_T,
    )
    rotvec = (
        matrix_to_rotvec(rel.reshape(-1, 3, 3)).reshape(*rotations.shape[:2], 3).cpu()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_soma_npz(
        str(output),
        rotvec,
        translations / 100.0,
        joint_names=list(soma.rig_data["joint_names"]),
        identity_model_type="mhr",
        identity_coeffs=shape,
        scale_params=params[:, 136:],
        joint_orient=soma._t_pose_orient,
        unit="meters",
        keep_root=False,
        extra_arrays={
            "bone_length_flexibles": params[:, 130:136],
            "preserve_frame_identity": np.asarray(True),
        },
    )
    return {
        "frames": count,
        "mean_vertex_error_m": float(error.mean() / 100.0),
        "max_vertex_error_m": float(error.max() / 100.0),
        "p95_vertex_error_m": float(np.percentile(error, 95) / 100.0),
        "mhr_model_sha256": file_sha256(model_path),
        "soma_neutral_sha256": file_sha256(asset_root / "SOMA_neutral.npz"),
        "identity_mode": "per_frame_preserved",
        "inversion_low_lod": inversion_low_lod,
        "world_transform": transform.tolist(),
    }
