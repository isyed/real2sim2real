"""Explicit phase execution for inspectable, resource-bounded reconstruction."""

import argparse
import json
from pathlib import Path
from real2sim2real.artifacts import write_json, publish


def execute(config, phase, *, check=False):
    from .capture import load

    out = Path(config["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = {
        "calibrate": "calibration.json",
        "prepare": "rgb",
        "camera": "camera_motion.npz",
        "publish": "artifact.json",
    }
    if phase in protected and (out / protected[phase]).exists():
        raise FileExistsError(
            f"Keep existing {phase} results; select a new output directory"
        )
    if phase == "calibrate":
        from .stereo import refine_calibration

        capture = load(config["capture"])
        result = refine_calibration(
            capture,
            config["calibration"]["fit_frames"],
            config["calibration"]["validation_frames"],
        )
        write_json(out / "calibration.json", result)
        if not result["accepted"]:
            raise ValueError(
                "Calibration rejected; retain original and inspect the report"
            )
    elif phase == "prepare":
        from .prepare import prepare

        prepare(
            load(config["capture"]),
            out,
            config["frame_count"],
            json.loads((out / "calibration.json").read_text()),
            image_scale=config.get("image_scale", 0.5),
        )
        write_json(out / "prompts.json", config["prompts"])
    elif phase in ("segment", "mesh", "track"):
        if phase == "mesh" and config.get("reuse_mesh_from") and not check:
            from .reuse import reuse_mesh

            options = config.get("model_options", {})
            reuse_mesh(
                config["reuse_mesh_from"],
                out,
                frame=options.get("mesh_frame", 0),
                object_id=options.get("object_id", 1),
            )
        else:
            from .runtime import run_model

            run_model(config, out, phase, check=check)
    elif phase == "camera":
        from .odometry import estimate

        estimate(out, **config.get("odometry", {}))
    elif phase == "publish":
        import numpy as np

        timeline = json.loads((out / "timeline.json").read_text())
        objects = np.load(out / "object_motion.npz")
        camera = np.load(out / "camera_motion.npz")
        if not objects["accepted"].all() or not camera["accepted"].all():
            raise ValueError("Cannot publish incomplete accepted reconstruction")
        poses = camera["world_from_camera"] @ objects["camera_from_object"]
        np.savez_compressed(
            out / "geometry.npz",
            world_from_object=poses,
            world_from_camera=camera["world_from_camera"],
            intrinsics=timeline["intrinsics"],
            frame_ids=timeline["frame_ids"],
            timestamps_s=timeline["timestamps_s"],
            accepted=objects["accepted"],
        )
        publish(
            out,
            "object-reconstruction",
            {
                "geometry": out / "geometry.npz",
                "mesh": out / "mesh/metric.glb",
                "timeline": out / "timeline.json",
                "camera_quality": out / "camera_motion_quality.json",
                "tracking_quality": out / "tracking_quality.json",
            },
            metadata=dict(
                units="metres",
                world_frame="first_target_camera",
                hand_reconstruction=False,
                capture_sha256=timeline["capture_sha256"],
            ),
        )
    else:
        raise ValueError("Unknown phase")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--phase",
        choices=[
            "calibrate",
            "prepare",
            "segment",
            "camera",
            "mesh",
            "track",
            "publish",
        ],
        required=True,
    )
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.check and a.phase not in ("segment", "mesh", "track"):
        p.error("--check is only for model imports")
    execute(config, a.phase, check=a.check)


if __name__ == "__main__":
    main()
