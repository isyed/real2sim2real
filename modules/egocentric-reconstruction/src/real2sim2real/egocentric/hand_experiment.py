"""Controlled hand-observation experiments in an immutable calibrated scene."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory


def subset_archive(path, output, frame_ids, times):
    with np.load(path, allow_pickle=False) as data:
        indices = np.searchsorted(data["frame_ids"], frame_ids)
        if (indices >= len(data["frame_ids"])).any() or not np.array_equal(
            data["frame_ids"][indices], frame_ids
        ):
            raise ValueError("Baseline does not contain the requested source frames")
        if not np.allclose(data["timestamps_s"][indices], times, atol=1e-8, rtol=0):
            raise ValueError("Baseline capture timestamps differ")
        n = len(data["frame_ids"])
        # Explicit static fields prevent accidental slicing when frame count is two.
        static = {"faces", "scale_per_hand", "fixed_betas"}
        selected = {
            key: data[key][indices]
            if key not in static and data[key].ndim and len(data[key]) == n
            else data[key]
            for key in data.files
        }
    np.savez_compressed(output, **selected)


def prepare(config):
    root, parent = Path(config["output"]), Path(config["baseline"])
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    out = fresh_directory(root / "inputs")
    for name in [
        "support/support.json",
        "hands_raw/shape.json",
        "hands_metric/quality.json",
    ]:
        destination = out / Path(name).name
        shutil.copyfile(parent / name, destination)
    subset_archive(
        parent / "hands2d/landmarks.npz",
        out / "landmarks.npz",
        source.frame_ids,
        source.times,
    )
    baseline = fresh_directory(root / "baseline" / "hands_raw")
    subset_archive(
        parent / "hands_raw/predictions.npz",
        baseline / "predictions.npz",
        source.frame_ids,
        source.times,
    )
    subset_archive(
        parent / "hands_metric/hands.npz",
        out / "archived_baseline.npz",
        source.frame_ids,
        source.times,
    )
    source.receipt(
        out,
        mask_ids=[config.get("object_id", 1), *config.get("mask_ids", [2, 3])],
        config=config,
    )
    write_json(
        out / "experiment.json",
        dict(
            schema="real2sim2real.hand-experiment/1",
            frame_ids=source.frame_ids.tolist(),
            timestamps_s=source.times.tolist(),
            baseline=str(parent),
            baseline_files={
                name: digest(parent / name)
                for name in [
                    "hands_raw/predictions.npz",
                    "hands_metric/hands.npz",
                    "hands2d/landmarks.npz",
                    "support/support.json",
                ]
            },
            shared_intrinsics=source.k.tolist(),
            hand_shape_fixed=True,
            hand_scale_fixed=True,
            camera_object_support_fixed=True,
            baseline_re_evaluation="same new partial-observation fitter; archived accepted geometry retained separately",
        ),
    )


def runtime_config(config, variant):
    root = Path(config["output"]).resolve()
    phase = {"detector": "detect-hands", "hamer": "hands", "hawor": "hawor-hands"}[
        variant
    ]
    out = root / ("inputs" if variant == "detector" else variant)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(config, output=str(out))
    settings["model_options"] = dict(
        reconstruction=config["reconstruction"],
        frame_start=config["frame_start"],
        frame_count=config["frame_count"],
        mask_ids=config.get("mask_ids", [2, 3]),
        hand_tracks=str(root / "inputs/hand_tracks/tracks.npz"),
        fixed_shape_from=str(root / "inputs/shape.json"),
        hand_crop_factor=config.get("hand_crop_factor", 2.0),
    )
    settings["read_only_inputs"] = [config["reconstruction"]]
    if variant != "detector":
        settings["read_only_inputs"].append(str(root / "inputs"))
    # Evaluation annotations live outside these mounts and never enter inference.
    return settings, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=[
            "prepare",
            "review-inputs",
            "detector",
            "hamer",
            "hawor",
            "fit",
            "evaluate",
        ],
        required=True,
    )
    parser.add_argument("--variant", choices=["baseline", "hamer", "hawor"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model_phase = args.phase in ("detector", "hamer", "hawor")
    if args.check and not model_phase:
        parser.error("--check is only for model import checks")
    if args.phase == "fit" and args.variant is None:
        parser.error("--variant is required for fitting")
    if not model_phase and not args.cpu_worker:
        from real2sim2real.bounded_process import run

        seconds = config.get("timeouts", {}).get(
            args.phase, 2400 if args.phase == "fit" else 300
        )
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not 0 < seconds <= 28800
        ):
            raise ValueError(
                "CPU stages require a finite deadline of at most eight hours"
            )
        command = [
            sys.executable,
            "-m",
            __name__,
            "--config",
            str(args.config.resolve()),
            "--phase",
            args.phase,
            "--cpu-worker",
        ]
        # __name__ is '__main__' for -m; use the installed module explicitly.
        command[2] = "real2sim2real.egocentric.hand_experiment"
        if args.variant:
            command += ["--variant", args.variant]
        return run(
            command,
            timeout=seconds,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
        )
    write_json(
        Path(config["output"])
        / "attempts"
        / args.phase
        / str(time.time_ns())
        / "source.json",
        dict(
            config=config,
            phase=args.phase,
            variant=args.variant,
            check_only=args.check,
            files={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        ),
    )
    if args.phase == "prepare":
        prepare(config)
    elif args.phase == "review-inputs":
        from .hand_review_inputs import prepare_review_inputs

        prepare_review_inputs(config)
    elif args.phase in ("detector", "hamer", "hawor"):
        from .runtime import run_model

        settings, phase = runtime_config(config, args.phase)
        run_model(settings, settings["output"], phase, check=args.check)
    elif args.phase == "fit":
        if args.variant is None:
            parser.error("--variant is required for fitting")
        from .hand_observation_fit import fit_sequence

        fit_sequence(config, args.variant)
    else:
        from .hand_experiment_review import evaluate

        evaluate(config)


if __name__ == "__main__":
    main()
