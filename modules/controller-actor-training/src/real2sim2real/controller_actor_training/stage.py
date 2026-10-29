"""FlashCHORD + FlashSAC training from an explicit, portable robot reference."""

import json
import os
from pathlib import Path
import subprocess
import sys

from real2sim2real.artifacts import load, role_path, publish, digest, write_json
from real2sim2real.stage import option_path, request_main


def overrides(request):
    options = request["options"]
    steps = options.get("environment_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("Set an explicit positive environment_steps training budget")
    worlds = options.get("worlds", 256)
    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise ValueError("worlds must be positive")
    # Never silently exceed the caller's transition budget when batching worlds.
    steps = (steps // worlds) * worlds
    if steps == 0:
        raise ValueError("environment_steps must cover at least one collection")
    checkpoint = options.get("checkpoint_every", 5000000)
    if (
        isinstance(checkpoint, bool)
        or not isinstance(checkpoint, int)
        or checkpoint <= 0
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    source = Path(request["input"])
    output = Path(request["output"])
    values = [
        f"experiment={options.get('experiment', 'g1_bounded_flash_sac')}",
        "task.parquet=" + json.dumps(str(role_path(source, "reference"))),
        "output_dir=" + json.dumps(str(output / "training")),
        "hydra.run.dir=" + json.dumps(str(output / "hydra")),
        "logging.mode=disabled",
        f"scene.world_count={worlds}",
        f"training.total_environment_steps={steps}",
        f"training.checkpoint.save_interval_environment_steps={checkpoint}",
    ]
    extra = options.get("overrides", [])
    if not isinstance(extra, list) or not all(
        isinstance(x, str) and "=" in x for x in extra
    ):
        raise ValueError("Hydra overrides must be a list of key=value strings")
    protected = (
        "task.parquet",
        "output_dir",
        "hydra.run.dir",
        "training.total_environment_steps",
        "logging.mode",
        "scene.world_count",
        "training.world_count",
    )
    if any(x.lstrip("+~").split("=", 1)[0] in protected for x in extra):
        raise ValueError(
            "Use explicit stage options for reference, budget, output and logging"
        )
    return values + extra


def run(request):
    source = Path(request["input"])
    output = Path(request["output"])
    options = request["options"]
    load(source, kinds={"robot-reference"})
    reference = load(source)
    if reference["metadata"].get("robot") != "g1" and options.get(
        "experiment", "g1_bounded_flash_sac"
    ).startswith("g1_"):
        raise ValueError("Select an embodiment/action recipe matching this robot")
    assets = option_path(request, "assets")
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    env = dict(
        os.environ,
        R2S2R_ASSETS=str(assets),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        WANDB_MODE="disabled",
    )
    command = [
        sys.executable,
        "-m",
        "real2sim2real.controller_actor_training.train",
        *overrides(request),
    ]
    write_json(output / "command.json", command)
    subprocess.run(command, env=env, check=True)
    checkpoints = sorted((output / "training").rglob("*.safetensors"))
    actors = [p for p in checkpoints if p.name.startswith("policy_")]
    if not actors:
        raise ValueError("Training returned without an actor checkpoint")
    files = {f"actor_{i}": p for i, p in enumerate(actors)}
    # Resume state/replay are retained by the trainer but never mistaken for an actor.
    files["configuration"] = output / "command.json"
    publish(
        output,
        "controller-policy",
        files,
        metadata=dict(
            algorithm="FlashSAC",
            simulator="FlashCHORD",
            parent_artifact_sha256=digest(source / "artifact.json"),
            requested_environment_steps=options["environment_steps"],
            actual_budget_environment_steps=options["environment_steps"]
            // options.get("worlds", 256)
            * options.get("worlds", 256),
            physical_deployment_approved=False,
            unassisted_evaluation_required=True,
        ),
    )


if __name__ == "__main__":
    request_main(run)
