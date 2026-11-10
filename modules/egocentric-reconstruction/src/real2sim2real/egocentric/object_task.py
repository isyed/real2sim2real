"""Export a calibrated object/table task without a human finger reference."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory


def prepare(config):
    source = Reconstruction(
        config["reconstruction"],
        config["frame_count"],
        start=config.get("frame_start", 0),
    )
    support_path = Path(config["support"])
    support = json.loads(support_path.read_text())
    if (
        not support["accepted"]
        or support["units"] != "metres"
        or support["up_axis"] != "+Z"
    ):
        raise ValueError("Accepted metric upright support required")
    if support["source_geometry_sha256"] != digest(source.path("geometry.npz")):
        raise ValueError("Support and reconstruction geometry differ")
    transforms = np.asarray(support["upright_from_world"]) @ source.world_from_object
    output = fresh_directory(config["output"])
    shutil.copyfile(source.path("mesh/metric.glb"), output / "object.glb")
    np.savez_compressed(
        output / "object_reference.npz",
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
        world_from_object=transforms,
    )
    task = dict(
        schema="real2sim2real.object-task/1",
        units="metres",
        up_axis="+Z",
        object=dict(
            mesh="object.glb",
            sha256=digest(output / "object.glb"),
            world_from_object=transforms[0].tolist(),
        ),
        table=dict(
            boundary_xy=support["observed_boundary_xy"],
            top_z_m=0.0,
            thickness_m=support["collision_thickness_m"],
            extent_assumption=support["boundary_kind"],
        ),
        goal=config["goal"],
        source_frame_ids=source.frame_ids.tolist(),
        human_fingers_used=False,
        reconstructed_object_motion_is_a_reference_only=True,
        sources={
            str(support_path): digest(support_path),
            str(source.path("geometry.npz")): digest(source.path("geometry.npz")),
            str(source.path("timeline.json")): digest(source.path("timeline.json")),
        },
    )
    write_json(output / "scene.json", task)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
