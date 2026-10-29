"""Versioned, self-contained numeric bundles; no pickle or external asset links."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .contracts import BundleData, COORDINATES, GEOMETRY_KEYS, OBSERVATION_KEYS, SCHEMA, ReconstructionBundle


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        name = stream.name
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            os.unlink(name)
            raise
    os.replace(name, path)


def arrays_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        name = stream.name
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            os.unlink(name)
            raise
    os.replace(name, path)


def arrays_read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if any(a.dtype.kind not in "biuf" for a in result.values()):
        raise ValueError(f"Only real numeric arrays are supported: {path.name}")
    return result


def _rigid_rotations(value: np.ndarray, name: str) -> None:
    # Historical packed trajectories contain small accumulated numerical drift
    # (observed determinant roundoff up to 3.96e-5). Preserve these values rather than
    # silently normalizing. Larger nonrigidity/reflections remain invalid.
    if not np.allclose(value @ value.swapaxes(-1, -2), np.eye(3), atol=1e-4, rtol=0):
        raise ValueError(f"{name} is not orthonormal")
    if not np.allclose(np.linalg.det(value), 1, atol=1e-4, rtol=0):
        raise ValueError(f"{name} determinant deviates from +1 by more than 1e-4")


def geometry_validate(g: dict[str, np.ndarray]) -> None:
    if set(g) != GEOMETRY_KEYS:
        raise ValueError(f"Geometry keys differ from contract: {sorted(set(g) ^ GEOMETRY_KEYS)}")
    if any(not np.isfinite(a).all() for a in g.values()):
        raise ValueError("Geometry contains nonfinite values")
    if any(a.dtype.kind != "f" for key, a in g.items() if key not in ("faces", "frame_ids")):
        raise ValueError("Geometry values other than face/frame indices must use floating-point arrays")
    ids, ts = g["frame_ids"], g["timestamps_s"]
    n = len(ids)
    if ids.shape != (n,) or n < 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("frame_ids must be nonempty, unique and increasing integers")
    if ts.shape != (n,) or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps_s must be increasing seconds, one per frame")
    v, f = g["vertices"], g["faces"]
    if v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3:
        raise ValueError("vertices must be [V,3]")
    if f.ndim != 2 or f.shape[1:] != (3,) or len(f) < 1 or f.dtype.kind not in "iu":
        raise ValueError("faces must be integer [F,3]")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("Mesh face index out of range")
    for key, shape in [("object_rotation", (n, 3, 3)), ("object_translation", (n, 3)),
                       ("intrinsics", (n, 3, 3)), ("world_to_camera", (n, 4, 4))]:
        if g[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    if g["object_scale"].shape != () or g["object_scale"] <= 0:
        raise ValueError("object_scale must be a positive scalar; deforming meshes are unsupported")
    _rigid_rotations(g["object_rotation"], "object_rotation")
    w = g["world_to_camera"]
    _rigid_rotations(w[:, :3, :3], "world_to_camera")
    if not np.allclose(w[:, 3], [0, 0, 0, 1]):
        raise ValueError("Invalid camera homogeneous transform")
    k = g["intrinsics"]
    if not np.allclose(k[:, 2], [0, 0, 1]) or np.any(k[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Invalid pinhole intrinsics")
    if not np.allclose(k[:, 1, 0], 0):
        raise ValueError("Intrinsics must use pinhole upper-triangular convention")


def observations_validate(o: dict[str, np.ndarray], n: int) -> None:
    if set(o) - OBSERVATION_KEYS:
        raise ValueError(f"Unknown observation arrays: {sorted(set(o) - OBSERVATION_KEYS)}")
    if any(not np.isfinite(a).all() for a in o.values()):
        raise ValueError("Observations contain nonfinite values; use reliability masks, not NaN")
    for key in ("reliable", "locked", "axis_facing"):
        if key in o and o[key].shape != (n,):
            raise ValueError(f"{key} must have shape [T]")
    for key in ("reliable", "locked", "object_mask", "occluder_mask", "contact_active"):
        if key in o and not np.isin(o[key], [0, 1]).all():
            raise ValueError(f"{key} must contain binary values")
    if "object_mask" in o:
        m = o["object_mask"]
        if m.ndim != 3 or m.shape[0] != n or min(m.shape[1:]) < 2:
            raise ValueError("object_mask must be [T,H,W]")
        if "reliable" not in o:
            raise ValueError("Masks require explicit per-frame reliability")
    if "occluder_mask" in o and ("object_mask" not in o or o["occluder_mask"].shape != o["object_mask"].shape):
        raise ValueError("occluder_mask must match object_mask")
    if "axis_facing" in o and not np.isin(o["axis_facing"], [-1, 0, 1]).all():
        raise ValueError("axis_facing must be -1 (away), 0 (unknown), or +1 (toward camera)")
    if "hand_points_world" in o:
        p = o["hand_points_world"]
        if p.ndim != 3 or p.shape[0] != n or p.shape[2] != 3 or p.shape[1] < 1:
            raise ValueError("hand_points_world must be [T,P,3]")
        if "contact_active" not in o or o["contact_active"].shape != p.shape[:2]:
            raise ValueError("Hand points require contact_active [T,P]")
    elif "contact_active" in o:
        raise ValueError("contact_active requires hand points")
    for key in ("human_reliable", "human_locked", "human_fit_frames"):
        if key in o and (o[key].shape != (n,) or not np.isin(o[key], [0, 1]).all()):
            raise ValueError(f"{key} must be binary [T]")
    if "human_mask" in o:
        m = o["human_mask"]
        if m.ndim != 3 or m.shape[0] != n or min(m.shape[1:]) < 2 or not np.isin(m, [0, 1]).all():
            raise ValueError("human_mask must be binary [T,H,W]")
        if "human_reliable" not in o:
            raise ValueError("Human masks require explicit reliability")
        if "object_mask" in o and m.shape != o["object_mask"].shape:
            raise ValueError("Human and object masks must share the intrinsics pixel grid")
    if "human_occluder_mask" in o and ("human_mask" not in o or o["human_occluder_mask"].shape != o["human_mask"].shape or
                                      not np.isin(o["human_occluder_mask"], [0, 1]).all()):
        raise ValueError("Human occluder mask must match the human mask")
    if "human_keypoints_xy" in o or "human_keypoints_confidence" in o:
        if o.get("human_keypoints_xy", np.array([])).shape != (n, 17, 2) or o.get("human_keypoints_confidence", np.array([])).shape != (n, 17):
            raise ValueError("Human keypoints require COCO17 xy[T,17,2] and confidence[T,17]")
        if np.any((o["human_keypoints_confidence"] < 0) | (o["human_keypoints_confidence"] > 1)):
            raise ValueError("Human confidence must be in [0,1]")
    if "human_contact_active" in o and (o["human_contact_active"].shape != (n, 2) or not np.isin(o["human_contact_active"], [0, 1]).all()):
        raise ValueError("human_contact_active must be binary [T,2], left/right order")


def asset_path(root: Path, spec: dict[str, str]) -> Path:
    relative = Path(spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset paths must remain inside the bundle")
    path = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Bundle assets cannot be symbolic links")
    if not path.is_file() or file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Missing or corrupt asset: {relative}")
    return path


def bundle_read(root: Path) -> ReconstructionBundle:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest cannot be a symbolic link")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("coordinates") != COORDINATES:
        raise ValueError("Unsupported schema or coordinate conventions; convert explicitly before importing")
    if not isinstance(manifest.get("sequence_id"), str) or not manifest["sequence_id"]:
        raise ValueError("sequence_id is required")
    if not isinstance(manifest.get("world_frame_id"), str) or not manifest["world_frame_id"]:
        raise ValueError("world_frame_id is required to distinguish coordinate frames across projects")
    assets = manifest["assets"]
    if "geometry" not in assets or set(assets) - {"geometry", "human", "observations", "repair_report", "baseline_geometry", "baseline_human"}:
        raise ValueError("Invalid asset roles")
    paths = {key: asset_path(root, spec) for key, spec in assets.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Asset roles must have distinct paths")
    geometry = arrays_read(paths["geometry"])
    geometry_validate(geometry)
    observations = arrays_read(paths["observations"]) if "observations" in paths else {}
    observations_validate(observations, len(geometry["frame_ids"]))
    if "human" in paths:
        human = arrays_read(paths["human"])
        if not human or any(not np.isfinite(a).all() for a in human.values()):
            raise ValueError("Human parameters must be finite numeric arrays")
        if not manifest.get("human", {}).get("encoding"):
            raise ValueError("Human payload requires an explicit encoding description")
    evidence = manifest.get("evidence", {})
    profile = manifest.get("object_profile", {})
    if "plane_normal_local" in profile:
        axis = np.asarray(profile["plane_normal_local"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
            raise ValueError("plane_normal_local must be a finite nonzero 3-vector")
    if "hand_points_world" in observations and not evidence.get("hand_points_source"):
        raise ValueError("Hand point provenance is required (predicted or annotated, not ground truth by default)")
    if "semantic_axis_local" in evidence:
        axis = np.asarray(evidence["semantic_axis_local"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
            raise ValueError("semantic_axis_local must be a finite, nonzero 3-vector")
    if "scale" in evidence:
        scale = evidence["scale"]
        if (scale.get("axis") not in (0, 1, 2) or not np.isfinite(scale.get("length_m", 0))
                or scale.get("length_m", 0) <= 0 or not scale.get("source")):
            raise ValueError("Scale evidence requires axis, positive length_m and source")
    return ReconstructionBundle(root, manifest, geometry, observations)


def bundle_create(root: Path, data: BundleData) -> ReconstructionBundle:
    """Create a new bundle from explicit conventions; never guess camera/world frames."""
    root = Path(root)
    metadata, geometry, observations, human = data.metadata, data.geometry, data.observations, data.human
    if root.exists():
        raise ValueError(f"Output already exists: {root}")
    geometry_validate(geometry)
    observations_validate(observations or {}, len(geometry["frame_ids"]))
    root.mkdir(parents=True)
    manifest = {**metadata, "schema": SCHEMA, "assets": {}}
    for key, values in [("geometry", geometry), ("observations", observations), ("human", human)]:
        if values is not None:
            path = root / f"{key}.npz"
            arrays_atomic(path, values)
            manifest["assets"][key] = {"path": path.name, "sha256": file_sha256(path)}
    json_atomic(root / "manifest.json", manifest)
    return bundle_read(root)


def bundle_copy(source: Path, destination: Path) -> ReconstructionBundle:
    """Copy only declared verified assets; a consumer gets no local links."""
    bundle = bundle_read(source)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Export destination must not already exist")
    destination.mkdir(parents=True)
    for spec in bundle.manifest["assets"].values():
        target = destination / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset_path(bundle.root, spec), target)
    shutil.copyfile(bundle.root / "manifest.json", destination / "manifest.json")
    return bundle_read(destination)
