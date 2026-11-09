"""Build support, articulated hands and diagnostics from an object reconstruction."""

import argparse
import json
from pathlib import Path
import time

from real2sim2real.artifacts import digest, write_json


def execute(config, phase):
    from .interaction_io import Reconstruction

    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "inputs.json").exists():
        Reconstruction(config["reconstruction"], config.get("frame_count")).receipt(
            root,
            mask_ids=[
                config.get("object_id", 1),
                *config["hand_landmarks"].get("mask_ids", [2, 3]),
            ],
            config=config,
        )
    # Each invocation preserves the exact CPU implementation and configuration.
    attempt = root / "attempts" / ("interaction-" + phase) / str(time.time_ns())
    write_json(
        attempt / "source.json",
        dict(
            config=config,
            files={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            input_receipt_sha256=digest(root / "inputs.json"),
        ),
    )
    if phase == "support":
        from .support import reconstruct_support

        return reconstruct_support(config)
    if phase == "hands2d":
        from .hand_landmarks import detect

        return detect(config)
    if phase in ("hands", "hands-check"):
        from .runtime import run_model

        config = dict(config)
        config["model_options"] = dict(
            config.get("model_options", {}),
            reconstruction=str(Path(config["reconstruction"]).resolve()),
            frame_count=config.get("frame_count"),
        )
        config["read_only_inputs"] = list(
            dict.fromkeys(
                [
                    str(Path(config["reconstruction"]).resolve()),
                    *config.get("read_only_inputs", []),
                ]
            )
        )
        return run_model(
            config, config["output"], "hands", check=phase == "hands-check"
        )
    if phase == "align-hands":
        from .metric_hands import reconstruct_hands

        return reconstruct_hands(config)
    if phase == "contacts":
        from .contact_diagnostics import diagnose

        return diagnose(config)
    if phase == "score":
        from .interaction_summary import summarize

        return summarize(config)
    if phase == "review":
        from .interaction_review import review

        return review(config)
    if phase == "publish":
        from .interaction_bundle import publish_scene

        return publish_scene(config)
    raise ValueError(f"Unknown interaction phase: {phase}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "support",
            "hands2d",
            "hands",
            "hands-check",
            "align-hands",
            "contacts",
            "score",
            "review",
            "publish",
        ],
        required=True,
    )
    args = parser.parse_args()
    execute(json.loads(args.config.read_text()), args.phase)


if __name__ == "__main__":
    main()
