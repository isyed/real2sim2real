"""Read-only reconstruction inputs for metric interaction reconstruction."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim2real.artifacts import digest, load, write_json
from .camera import rigid


class Reconstruction:
    def __init__(self, root, count=None, *, start=0):
        self.root = Path(root).resolve()
        artifact = load(self.root, kinds={"object-reconstruction"})
        if artifact["metadata"].get("units") != "metres":
            raise ValueError("Interaction reconstruction requires metres")
        self.timeline = json.loads((self.root / "timeline.json").read_text())
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("Invalid interaction start index")
        self.start = start
        self.count = self.timeline["frames"] - start if count is None else int(count)
        if not 1 <= self.count <= self.timeline["frames"] - start:
            raise ValueError("Invalid interaction frame count")
        selected = slice(start, start + self.count)
        self.frame_ids = np.asarray(self.timeline["frame_ids"][selected])
        self.times = np.asarray(self.timeline["timestamps_s"][selected])
        self.k = np.asarray(self.timeline["intrinsics"], dtype=float)
        with np.load(self.root / "geometry.npz") as geometry:
            if not geometry["accepted"][selected].all():
                raise ValueError("Require accepted camera and object poses")
            self.world_from_camera = geometry["world_from_camera"][selected].copy()
            self.world_from_object = geometry["world_from_object"][selected].copy()
            if not np.array_equal(geometry["frame_ids"][selected], self.frame_ids):
                raise ValueError("Geometry frame IDs differ")
            if not np.allclose(
                geometry["timestamps_s"][selected], self.times, atol=1e-8, rtol=0
            ):
                raise ValueError("Geometry timestamps differ")
        for matrix in [*self.world_from_camera, *self.world_from_object]:
            rigid(matrix)
        if self.count > 1 and not (np.diff(self.times) > 0).all():
            raise ValueError("Require increasing capture timestamps")

    def path(self, name):
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Inputs must be real files within the reconstruction")
        return path

    def rgb(self, i):
        return np.asarray(
            Image.open(self.path(f"rgb/{i + self.start:06d}.png")).convert("RGB")
        )

    def depth(self, i):
        return np.load(self.path(f"depth/{i + self.start:06d}.npy"))

    def mask(self, i, identifier):
        return (
            np.asarray(
                Image.open(self.path(f"masks/{identifier}/{i + self.start:06d}.png"))
            )
            > 0
        )

    def receipt(self, output, *, mask_ids, config):
        names = ["artifact.json", "geometry.npz", "timeline.json", "mesh/metric.glb"]
        for i in range(self.start, self.start + self.count):
            names += [f"rgb/{i:06d}.png", f"depth/{i:06d}.npy"]
            names += [f"masks/{j}/{i:06d}.png" for j in mask_ids]
        write_json(
            Path(output) / "inputs.json",
            dict(
                reconstruction=str(self.root),
                config=config,
                files={name: digest(self.path(name)) for name in names},
                ground_truth_used=False,
            ),
        )


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
