"""Decode retained MHR parameters into a human surface for inspection.

This does not estimate a new pose or repair the input. Optional Torch is loaded
only during decoding. Outputs use the same OpenCV camera/world frame, in metres,
as reconstruction geometry, with source/model hashes and original frame order.
"""
from pathlib import Path
import json
import numpy as np

from .human import CV_NATIVE, _frames, file_sha256, rigid_transform


def native_to_world(vertices_cm, transform):
    vertices = np.asarray(vertices_cm)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite native human vertices [T,V,3]")
    matrix = rigid_transform(transform)
    camera_m = vertices @ CV_NATIVE / 100.0
    return (camera_m @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def decode_surface(human_file, model_file, output, *, batch_size=8, world_transform=None):
    """CPU-only, bounded-memory surface decoding; refuse partial-output reuse."""
    import torch

    human_file, model_file, output = map(Path, (human_file, model_file, output))
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
        raise ValueError("Surface decoding batch size must be 1..64")
    transform = rigid_transform(np.eye(4) if world_transform is None else world_transform)
    identity = dict(human_sha256=file_sha256(human_file), model_sha256=file_sha256(model_file),
                    world_transform=transform.tolist())
    if output.exists():
        receipt = json.loads((output / "surface.json").read_text())
        if any(receipt[k] != v for k, v in identity.items()):
            raise ValueError("Cached surface inputs changed")
        for key in ("vertices", "faces"):
            if file_sha256(output / f"{key}.npy") != receipt[f"{key}_sha256"]:
                raise ValueError("Cached human surface changed")
        return receipt
    with np.load(human_file, allow_pickle=False) as packed:
        params = np.asarray(packed["mhr_model_params"], dtype=np.float32)
        if params.ndim != 2 or params.shape[1] != 204 or not len(params) or not np.isfinite(params).all():
            raise ValueError("Expected finite MHR parameters [T,204]")
        count = len(params)
        shape = _frames(packed["shape_params"], count, 45, "shape").astype(np.float32)
        face = _frames(packed.get("face_params", np.zeros(72)), count, 72, "face").astype(np.float32)
    model = torch.jit.load(str(model_file), map_location="cpu").eval()
    faces = model.character_torch.mesh.faces.detach().cpu().numpy().astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0:
        raise ValueError("Invalid model topology")
    output.mkdir(parents=True)
    surface = None
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            vertices, _ = model(torch.from_numpy(shape[start:end]),
                                torch.from_numpy(params[start:end]), torch.from_numpy(face[start:end]))
            vertices = native_to_world(vertices.cpu().numpy(), transform)
            if surface is None:
                if faces.max() >= vertices.shape[1]:
                    raise ValueError("Topology exceeds vertex count")
                surface = np.lib.format.open_memmap(output / "vertices.npy", mode="w+",
                    dtype=np.float32, shape=(count, vertices.shape[1], 3))
            surface[start:end] = vertices
    surface.flush()
    np.save(output / "faces.npy", faces, allow_pickle=False)
    receipt = dict(schema="real2sim2real.human-surface/1", **identity, frames=count,
                   vertex_count=surface.shape[1], face_count=len(faces), units="metres",
                   coordinate_convention="opencv", new_pose_estimation=False, repair_applied=False,
                   vertices_sha256=file_sha256(output / "vertices.npy"),
                   faces_sha256=file_sha256(output / "faces.npy"))
    (output / "surface.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
