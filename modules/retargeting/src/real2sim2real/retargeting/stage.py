"""Convert a reconstructed human/object sequence into a robot reference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from real2sim2real.artifacts import load, role_path, publish, digest, write_json
from real2sim2real.stage import option_path, request_main


def run(request):
    source = Path(request["input"])
    out = Path(request["output"])
    options = request["options"]
    artifact = load(source, kinds={"reconstruction", "soma"})
    assets = option_path(request, "assets")
    configs = option_path(request, "robot_config_root")
    os.environ["R2S2R_ASSETS"] = str(assets)
    inputs = out / "retarget_input"
    inputs.mkdir()
    geometry = np.load(role_path(source, "geometry"), allow_pickle=False)
    n = len(geometry["frame_ids"])
    times = geometry["timestamps_s"]
    delta = np.diff(times)
    if n < 2 or not np.allclose(delta, np.median(delta), rtol=0.02, atol=1e-4):
        raise ValueError(
            "The current IK exporter requires a uniform source timeline; explicitly resample irregular videos first"
        )
    fps = 1 / float(np.median(delta))
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    poses[:, :3, :3] = geometry["object_rotation"]
    poses[:, :3, 3] = geometry["object_translation"]
    np.save(inputs / "poses.npy", poses)
    mesh_dir = inputs / "reconstructed_mesh"
    mesh_dir.mkdir()
    import trimesh

    mesh = trimesh.Trimesh(
        geometry["vertices"] * float(geometry["object_scale"]),
        geometry["faces"],
        process=False,
    )
    mesh.export(mesh_dir / "output_aligned.glb")
    if artifact["kind"] == "soma":
        shutil.copy2(role_path(source, "soma"), inputs / "soma_params.npz")
    else:
        from real2sim2real.human import invert_mhr_to_soma

        with np.load(role_path(source, "human"), allow_pickle=False) as data:
            packed = dict(data)
        diagnostics = invert_mhr_to_soma(
            packed,
            mhr_model=option_path(request, "mhr_model"),
            soma_assets=assets / "body_models/soma",
            output=inputs / "soma_params.npz",
            world_transform=np.eye(4),
            device=options.get("device", "cpu"),
        )
        write_json(out / "human_conversion.json", diagnostics)
    if options.get("support_plane"):
        shutil.copy2(
            option_path(request, "support_plane"), inputs / "ground_plane.json"
        )
    args = [
        sys.executable,
        "-m",
        "real2sim2real.retargeting.soma_to_robot",
        str(inputs),
        "--save",
        "--robot-name",
        options.get("robot", "g1"),
        "--robot-config-root",
        str(configs),
        "--soma-data-root",
        str(assets / "body_models/soma"),
        "--motion-root",
        str(out / "motion"),
        "--fps",
        str(fps),
        "--scale",
        str(options.get("human_to_robot_scale", 1.0)),
        "--contact-threshold",
        str(options.get("contact_threshold_m", 0.02)),
        "--preserve-frame-identity",
        "--device",
        options.get("device", "cpu"),
    ]
    write_json(out / "command.json", args)
    subprocess.run(args, check=True)
    references = list((out / "motion").rglob("*.parquet"))
    if len(references) != 1:
        raise ValueError("Expected exactly one retargeted reference")
    # Store a reference at the artifact root and make its object paths portable.
    from .motion_schema import load_motion_data_parquet, save_motion_parquet

    motion = load_motion_data_parquet(references[0])
    motion.sequence_id = artifact['metadata'].get('sequence_id', motion.sequence_id)
    for attr in ["object_mesh_paths", "object_urdf_paths"]:
        setattr(
            motion,
            attr,
            [
                str(Path(p).resolve().relative_to(out.resolve()))
                for p in getattr(motion, attr)
            ],
        )
    partition = save_motion_parquet(
        motion, out / "portable", partition_cols=[], file_name="reference.parquet"
    )
    shutil.copy2(partition / "reference.parquet", out / "reference.parquet")
    files = {"reference": out / "reference.parquet"}
    if options.get("support_asset"):
        # An optional explicit, already robot-aligned support surface. The
        # reference loader discovers this colocated USDA without dataset names.
        target = out / "support.usda"
        shutil.copy2(option_path(request, "support_asset"), target)
        files["support"] = target
    for index, p in enumerate(sorted(out.rglob("*"))):
        if (
            p.is_file()
            and p.suffix in [".obj", ".urdf", ".mtl", ".png", ".jpg", ".json"]
            and p.name not in ["artifact.json", "request.json"]
        ):
            files["asset_" + str(index)] = p
    publish(
        out,
        "robot-reference",
        files,
        metadata=dict(
            robot=options.get("robot", "g1"),
            fps=fps,
            frames=n,
            parent_artifact_sha256=digest(source / "artifact.json"),
            reference_contacts="geometric proximity; not measured physical contacts",
            physics_validated=False,
        ),
    )


if __name__ == "__main__":
    request_main(run)
