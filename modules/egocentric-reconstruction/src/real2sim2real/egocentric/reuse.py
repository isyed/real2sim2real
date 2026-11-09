"""Reuse a metric mesh only when its observed inputs are identical."""

import json
from pathlib import Path
import shutil

from real2sim2real.artifacts import digest, write_json


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
