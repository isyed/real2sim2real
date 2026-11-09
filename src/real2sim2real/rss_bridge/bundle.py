"""Explicit, reversible coordinate exchange for RGB-derived object poses."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, load, publish, write_json


def rigid(matrix, name="transform", *, rotation_atol=2e-6):
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite rigid 4x4 matrices")
    rotation = value[..., :3, :3]
    if (not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                               np.eye(3), atol=rotation_atol, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=rotation_atol, rtol=0)):
        raise ValueError(f"{name}: scale, reflection and nonrigid transforms are forbidden")
    return value


def read_geometry(path):
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key] for key in source.files}
    required = {"vertices", "faces", "object_rotation", "object_translation",
                "object_scale", "intrinsics", "world_to_camera", "frame_ids",
                "timestamps_s"}
    if missing := required - result.keys():
        raise ValueError(f"Missing explicit geometry fields: {sorted(missing)}")
    for key in required:
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite geometry: {key}")
    ids = result["frame_ids"]
    n = len(ids)
    if (ids.shape != (n,) or n == 0 or ids.dtype.kind not in "iu"
            or np.any(np.diff(ids) <= 0)):
        raise ValueError("Frame IDs must be unique, ordered integers")
    times = result["timestamps_s"]
    if times.shape != (n,) or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be strictly increasing seconds")
    for key, shape in {"object_rotation": (n, 3, 3),
                       "object_translation": (n, 3),
                       "world_to_camera": (n, 4, 4),
                       "intrinsics": (n, 3, 3)}.items():
        if result[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape; expected {shape}")
    if result["object_scale"].shape != () or float(result["object_scale"]) <= 0:
        raise ValueError("Object scale must be one explicit positive scalar")
    vertices, faces = result["vertices"], result["faces"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Expected mesh-local vertices")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.dtype.kind not in "iu" or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError("Expected valid triangle indices")
    k = result["intrinsics"]
    if (np.any(k[:, (0, 1), (0, 1)] <= 0)
            or not np.allclose(k[:, 2, :], [0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(k[:, 0, 1], 0, atol=1e-9, rtol=0)
            or not np.allclose(k[:, 1, 0], 0, atol=1e-9, rtol=0)):
        raise ValueError("RSS bridge requires positive focal lengths and zero-skew intrinsics")
    rigid(result["world_to_camera"], "world_to_camera")
    source_poses(result)
    return result


def source_poses(geometry):
    n = len(geometry["frame_ids"])
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = geometry["object_rotation"]
    poses[:, :3, 3] = geometry["object_translation"]
    # Historical float32 tracking matrices can accumulate a few ppm of
    # orthogonality drift. Preserve their bytes, but bound that drift before
    # making the proper rotations required by RSS's quaternion interface.
    return rigid(poses, "object poses", rotation_atol=1e-5)


def exchange_poses(geometry):
    poses = source_poses(geometry).copy()
    u, _, vt = np.linalg.svd(poses[:, :3, :3])
    poses[:, :3, :3] = u @ vt
    return rigid(poses)


def create_bundle(geometry, mesh, human, output, *, source_to_rss,
                  width, height, source_frame, rss_revision,
                  human_coordinate_frame, rgb=None):
    """Copy immutable inputs and export a full-frame object-pose edit table.

    Geometry uses metres, seconds, mesh-local vertices and CV cameras
    (+X right, +Y down, +Z forward). source_to_rss is a rigid world transform,
    never a scale correction. The human is an opaque, hash-preserved asset;
    this interface does not claim it shares the object coordinate frame.
    """
    source = read_geometry(geometry)
    transform = rigid(source_to_rss, "source_to_rss")
    if transform.shape != (4, 4):
        raise ValueError("One explicit source_to_rss world transform is required")
    if width <= 0 or height <= 0 or not source_frame or not human_coordinate_frame:
        raise ValueError("Image dimensions and coordinate-frame labels are required")
    if not rss_revision:
        raise ValueError("Pin the external RSS revision")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, source_path, name in [
        ("geometry", geometry, "geometry.npz"),
        ("source_mesh", mesh, "source_mesh" + Path(mesh).suffix),
        ("human", human, "human" + Path(human).suffix),
    ]:
        destination = output / name
        shutil.copyfile(source_path, destination)
        files[role] = destination
    if rgb is not None:
        files["rgb"] = output / ("source" + Path(rgb).suffix)
        shutil.copyfile(rgb, files["rgb"])
    # Export the actual trajectory's mesh-local geometry. The original visual
    # asset is retained separately; GLB scene-node transforms cannot alter it.
    import trimesh
    files["exchange_mesh"] = output / "object.ply"
    trimesh.Trimesh(vertices=source["vertices"], faces=source["faces"],
                    process=False).export(files["exchange_mesh"])
    preview_mesh = trimesh.load(files["exchange_mesh"], process=False)
    if (preview_mesh.vertices.shape != source["vertices"].shape
            or not np.array_equal(preview_mesh.faces, source["faces"])):
        raise ValueError("Mesh export changed vertex ordering or topology")
    mesh_error = float(np.max(np.linalg.norm(preview_mesh.vertices - source["vertices"], axis=1)))
    if mesh_error * float(source["object_scale"]) > 1e-7:
        raise ValueError("Mesh preview precision exceeds 0.1 micrometre; use a higher precision format")
    files["poses"] = output / "poses_rss.npz"
    np.savez_compressed(files["poses"], frame_ids=source["frame_ids"],
                        timestamps_s=source["timestamps_s"],
                        object_to_rss=transform @ exchange_poses(source))
    drift = source["object_rotation"].swapaxes(-1, -2) @ source["object_rotation"] - np.eye(3)
    delta = exchange_poses(source)[:, :3, :3] - source["object_rotation"]
    vertex_radius = float(np.max(np.linalg.norm(source["vertices"], axis=1))) * float(source["object_scale"])
    publish(output, "rss-source-exchange", files, metadata=dict(
        units="meters", time_units="seconds", source_frame=source_frame,
        source_to_rss=transform.tolist(), rss_world="explicit_rigid_source_transform",
        camera_basis="x_right_y_down_z_forward", width=int(width), height=int(height),
        rss_revision=rss_revision, human_coordinate_frame=human_coordinate_frame,
        human_preview_aligned=False, frame_count=len(source["frame_ids"]),
        usage="source_inspection_only", physical_validation=False,
        source_geometry_sha256=digest(geometry),
        exchange_mesh_max_vertex_error_m=mesh_error * float(source["object_scale"]),
        rss_rotation_representation="nearest_proper_rotation; source bytes retained",
        source_rotation_orthogonality_max_abs=float(np.max(np.abs(drift))),
        rss_rotation_vertex_displacement_bound_m=float(np.max(np.linalg.norm(delta, axis=(1, 2)))) * vertex_radius))
    return output


def import_poses(bundle, edited_poses, output):
    """Return a new candidate; never mutate or promote the exchange source.

    The only permitted edits are object rotation/translation. Camera, scale,
    mesh, human motion, complete frame IDs and timestamps remain unchanged.
    """
    bundle = Path(bundle)
    receipt = load(bundle, kinds={"rss-source-exchange"})
    paths = {role: bundle / row["path"] for role, row in receipt["files"].items()}
    geometry = read_geometry(paths["geometry"])
    with np.load(edited_poses, allow_pickle=False) as edited:
        if set(edited.files) != {"frame_ids", "timestamps_s", "object_to_rss"}:
            raise ValueError("Pose edits may contain only frame IDs, timestamps and poses")
        if (not np.array_equal(edited["frame_ids"], geometry["frame_ids"])
                or not np.array_equal(edited["timestamps_s"], geometry["timestamps_s"])):
            raise ValueError("Missing, reordered or retimed source frames")
        new = rigid(edited["object_to_rss"], "edited poses")
    with np.load(paths["poses"], allow_pickle=False) as original:
        old = original["object_to_rss"]
    if new.shape != old.shape:
        raise ValueError("Edited pose sequence must cover every source frame")
    changed = np.any(new != old, axis=(1, 2))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for role in ("geometry", "source_mesh", "human"):
        files[role] = output / paths[role].name
        shutil.copyfile(paths[role], files[role])
    if changed.any():
        inverse = np.linalg.inv(np.asarray(receipt["metadata"]["source_to_rss"]))
        source = rigid(inverse @ new)
        # Unedited frames are copied exactly, avoiding numerical churn.
        geometry["object_rotation"] = geometry["object_rotation"].copy()
        geometry["object_translation"] = geometry["object_translation"].copy()
        geometry["object_rotation"][changed] = source[changed, :3, :3]
        geometry["object_translation"][changed] = source[changed, :3, 3]
        np.savez_compressed(files["geometry"], **geometry)
    files["edited_poses"] = output / "edited_poses_rss.npz"
    shutil.copyfile(edited_poses, files["edited_poses"])
    files["edit_receipt"] = output / "edit_receipt.json"
    write_json(files["edit_receipt"], dict(
        exchange_manifest_sha256=digest(bundle / "artifact.json"),
        edited_poses_sha256=digest(edited_poses),
        changed_frame_ids=geometry["frame_ids"][changed].tolist(),
        identity_roundtrip=not bool(changed.any()),
        identity_geometry_byte_preserved=(digest(files["geometry"]) == digest(paths["geometry"])),
        max_translation_change_m=float(np.max(np.linalg.norm(new[:, :3, 3] - old[:, :3, 3], axis=1))),
        physical_validation=False, source_validation=False, selected=False))
    publish(output, "rss-object-pose-candidate", files, metadata=dict(
        parent_exchange_sha256=digest(bundle / "artifact.json"),
        units="meters", time_units="seconds", frame_count=len(new),
        source_frame=receipt["metadata"]["source_frame"],
        human_coordinate_frame=receipt["metadata"]["human_coordinate_frame"],
        physical_validation=False, source_validation=False, selected=False))
    return output

