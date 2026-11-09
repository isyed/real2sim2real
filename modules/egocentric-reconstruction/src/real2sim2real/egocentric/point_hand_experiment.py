"""Bounded point-observation experiment with isolated fitting and evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory
from .hand_point_tracks import select_seeds


def prepare(config):
    root, parent = Path(config["output"]), Path(config["parent"])
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    observations = dict(np.load(parent / "inputs/landmarks.npz"))
    for name in ("frame_ids", "timestamps_s"):
        expected = source.frame_ids if name == "frame_ids" else source.times
        if not np.array_equal(observations[name], expected):
            raise ValueError("Seed observations must match the source timeline")
    seed_ids, heldout_ids = config["seed_frame_ids"], config["heldout_frame_ids"]
    if not heldout_ids or not set(heldout_ids).issubset(set(source.frame_ids)):
        raise ValueError("Require held-out frame IDs within the input window")
    masks = np.asarray(
        [[source.mask(i, h) for h in config["mask_ids"]] for i in range(source.count)]
    )
    seeds = select_seeds(observations, masks, source.frame_ids, seed_ids, heldout_ids)
    out = fresh_directory(root / "point_inputs")
    np.save(out / "rgb.npy", np.asarray([source.rgb(i) for i in range(source.count)]))
    np.save(out / "masks.npy", masks)
    np.savez_compressed(
        out / "seeds.npz",
        **seeds,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "protocol.json",
        dict(
            schema="real2sim2real.point-hand-protocol/1",
            seed_frame_ids=seed_ids,
            heldout_frame_ids=heldout_ids,
            seeds_sha256=digest(out / "seeds.npz"),
            rgb_sha256=digest(out / "rgb.npy"),
            masks_sha256=digest(out / "masks.npy"),
            heldout_labels_used_for_seeds=False,
            queries=len(seeds["hand"]),
            point_limits=config.get("point_limits", {}),
            optimizer=config.get("optimizer", {}),
            hand_prediction_source="independent_2d_detector_only",
        ),
    )
    fitting = fresh_directory(root / "fit_inputs")
    files = {
        "raw.npz": parent / "hawor/hands_raw/predictions.npz",
        "raw_provenance.json": parent / "hawor/hands_raw/provenance.json",
        "baseline.npz": parent / "inputs/archived_baseline.npz",
        "crops.npz": parent / "inputs/hand_tracks/tracks.npz",
        "support.json": parent / "inputs/support.json",
    }
    for name, path in files.items():
        shutil.copyfile(path, fitting / name)
    write_json(
        fitting / "receipt.json", {name: digest(path) for name, path in files.items()}
    )
    source.receipt(
        fitting, mask_ids=[config["object_id"], *config["mask_ids"]], config=config
    )
    evaluation = fresh_directory(root / "evaluation_inputs")
    use = np.isin(source.frame_ids, heldout_ids)
    np.savez_compressed(
        evaluation / "automatic_landmarks.npz",
        keypoints_px=observations["keypoints_px"][use],
        accepted=observations["accepted"][use],
        frame_ids=source.frame_ids[use],
    )
    # Copy only review images/labels after preparing inference bundles. No model
    # worker mounts this directory or the parent experiment root.
    shutil.copytree(parent / "evaluation_inputs", evaluation / "views")
    write_json(
        root / "prepared.json",
        dict(
            config=config,
            source_files={str(p): digest(p) for p in files.values()},
            point_input_files={p.name: digest(p) for p in out.iterdir()},
            evaluation_only=True,
            control_ready=False,
        ),
    )


def runtime_config(config, phase):
    root = Path(config["output"]).resolve()
    tracking = phase == "track"
    name = "track-hand-points" if tracking else "refine-point-hands"
    out = root / ("tracking" if tracking else "refined")
    out.mkdir(parents=True, exist_ok=True)
    options = dict(
        point_inputs=str(root / "point_inputs"),
        point_limits=config.get("point_limits", {}),
        tracking_size=config.get("tracking_size", 512),
    )
    read_only = [str(root / "point_inputs")]
    weights = {"tapir": config["weights"]["tapir"]}
    if not tracking:
        options = dict(
            reconstruction=config["reconstruction"],
            frame_start=config["frame_start"],
            frame_count=config["frame_count"],
            fit_inputs=str(root / "fit_inputs"),
            point_tracks=str(root / "tracking/point_tracks"),
            diagnostic_frame_ids=config["heldout_frame_ids"],
            mask_ids=config["mask_ids"],
            object_id=config["object_id"],
            optimizer=config.get("optimizer", {}),
            contacts=config.get("contacts", {}),
        )
        read_only = [
            config["reconstruction"],
            str(root / "fit_inputs"),
            str(root / "tracking/point_tracks"),
        ]
        weights = {"hamer": config["weights"]["hamer"]}
    if phase == "sequence":
        name = "sequence-point-hands"
        directory = config.get("candidate_directory", "sequence")
        if (
            not directory
            or Path(directory).name != directory
            or directory
            in (".", "..", "refined", "tracking", "fit_inputs", "point_inputs")
        ):
            raise ValueError("Use a fresh sequence candidate directory name")
        out = root / directory
        out.mkdir(parents=True, exist_ok=True)
        options.update(
            sequence_parent=str(root / "refined/hands_metric"),
            anchor_priors=str(
                Path(config["parent"]) / "baseline/hands_raw/predictions.npz"
            ),
            sequence_optimizer=config.get("sequence_optimizer", {}),
        )
        read_only.extend([options["sequence_parent"], options["anchor_priors"]])
    return dict(
        config,
        output=str(out),
        weights=weights,
        read_only_inputs=read_only,
        model_options=options,
    ), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["prepare", "track", "refine", "sequence", "evaluate"],
        required=True,
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.check and args.phase not in ("track", "refine", "sequence"):
        parser.error("Checks apply only to model workers")
    if args.phase in ("track", "refine", "sequence"):
        from .runtime import run_model

        settings, phase = runtime_config(config, args.phase)
        return run_model(settings, settings["output"], phase, check=args.check)
    if not args.cpu_worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                "real2sim2real.egocentric.point_hand_experiment",
                "--config",
                str(args.config.resolve()),
                "--phase",
                args.phase,
                "--cpu-worker",
            ],
            timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
        )
    if args.phase == "prepare":
        prepare(config)
    else:
        from .point_hand_review import evaluate

        evaluate(config)


if __name__ == "__main__":
    main()
