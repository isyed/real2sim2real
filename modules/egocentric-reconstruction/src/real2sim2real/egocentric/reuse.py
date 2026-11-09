"""Reuse a metric mesh only when its observed inputs are identical."""

import json
from pathlib import Path
import shutil

from real2sim2real.artifacts import digest, write_json


def fork_tracking_inputs(source, destination):
    """Copy observed inputs and a fixed mesh into an isolated tracking hypothesis."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Tracking source and destination must not overlap")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Tracking hypotheses require an empty destination")
    names = [
        "rgb",
        "depth",
        "masks",
        "mesh",
        "timeline.json",
        "camera.json",
        "camera_motion.npz",
        "camera_motion_quality.json",
        "calibration.json",
        "prompts.json",
        "depth_quality.json",
    ]
    hashes = {}
    for name in names:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(path)
        files = list(path.rglob("*")) if path.is_dir() else [path]
        if path.is_symlink() or any(p.is_symlink() for p in files):
            raise ValueError("Tracking inputs must be real files, not symlinks")
        for file in files:
            if file.is_file():
                hashes[str(file.relative_to(source))] = digest(file)
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, destination / name)
        else:
            shutil.copy2(path, destination / name)
    if any(digest(destination / name) != value for name, value in hashes.items()):
        raise ValueError("Source inputs changed while copying; reject this hypothesis")
    write_json(
        destination / "tracking_inputs.json", dict(source=str(source), files=hashes)
    )


def reuse_mesh(source, destination, *, frame=0, object_id=1):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (destination / "mesh").exists():
        raise FileExistsError(
            "Keep the existing mesh; select a new hypothesis directory"
        )
    timelines = [
        json.loads((root / "timeline.json").read_text())
        for root in (source, destination)
    ]
    for key in ("capture_sha256", "intrinsics", "image_size"):
        if timelines[0][key] != timelines[1][key]:
            raise ValueError(f"Mesh reuse requires identical {key}")
    quality = json.loads((source / "mesh/input_quality.json").read_text())
    if quality["frame"] != frame:
        raise ValueError("Mesh conditioning frame differs")
    inputs = [
        f"rgb/{frame:06d}.png",
        f"depth/{frame:06d}.npy",
        f"masks/{object_id}/{frame:06d}.png",
    ]
    hashes = {}
    for name in inputs:
        hashes[name] = digest(source / name)
        if hashes[name] != digest(destination / name):
            raise ValueError(f"Mesh conditioning input changed: {name}")
    files = {
        str(p.relative_to(source / "mesh")): digest(p)
        for p in (source / "mesh").rglob("*")
        if p.is_file()
    }
    if "metric.glb" not in files:
        raise FileNotFoundError("Source metric mesh is missing")
    shutil.copytree(source / "mesh", destination / "mesh")
    write_json(
        destination / "mesh/reuse.json",
        dict(
            source=str(source),
            inputs=hashes,
            files=files,
            frame=frame,
            object_id=object_id,
            ground_truth_used=False,
        ),
    )
