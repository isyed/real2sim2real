"""A selected ordered stage chain; independently installed stage environments."""

from __future__ import annotations
import json
from importlib.metadata import entry_points
import os
from pathlib import Path
import sys

from .artifacts import digest, load, write_json

STAGES = {
    "reconstruction": (
        "real2sim2real.reconstruction.stage",
        {"video"},
        "reconstruction",
    ),
    "reconstruction-repair": (
        "real2sim2real.reconstruction_repair.stage",
        {"reconstruction"},
        "reconstruction",
    ),
    "retargeting": (
        "real2sim2real.retargeting.stage",
        {"reconstruction", "soma"},
        "robot-reference",
    ),
    "controller-actor-training": (
        "real2sim2real.controller_actor_training.stage",
        {"robot-reference"},
        "controller-policy",
    ),
}


def stage_registry():
    """Discover lightweight stage descriptors from separately installed modules."""
    stages = dict(STAGES)
    for entry in entry_points(group="real2sim2real.stages"):
        if entry.name in stages:
            raise ValueError("Duplicate stage registration: " + entry.name)
        module, accepts, produces = entry.load()()
        if not isinstance(module, str) or not accepts or not isinstance(produces, str):
            raise ValueError("Invalid stage descriptor: " + entry.name)
        stages[entry.name] = (module, set(accepts), produces)
    return stages


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema") != "real2sim2real.pipeline/1":
        raise ValueError("Expected real2sim2real.pipeline/1 configuration")
    if not isinstance(config.get("modules"), dict) or not isinstance(
        config.get("input"), str
    ):
        raise ValueError("Explicit input and module configurations required")
    return config, path.parent


def plan(config, base, *, selected=None):
    selected = selected or config.get("stages")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty stage list without duplicates")
    kind = config.get("input_kind", "video")
    previous = (base / config["input"]).resolve()
    output = (base / config["output"]).resolve()
    if output == previous or output.is_relative_to(previous) and previous.is_dir():
        raise ValueError("Pipeline outputs must be separate from input artifacts")
    rows = []
    stages = stage_registry()
    for index, name in enumerate(selected):
        if name not in stages:
            raise ValueError("Unknown stage: " + name)
        module, accepts, produces = stages[name]
        if kind not in accepts:
            raise ValueError(f"{name} accepts {sorted(accepts)}, received {kind}")
        options = config["modules"].get(name)
        if not isinstance(options, dict):
            raise ValueError("Missing module configuration: " + name)
        target = output / f"{index:02d}-{name}"
        execution = options.get("execution", {})
        timeout = execution.get("timeout_seconds", 10800)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("Each stage requires a positive time limit")
        python = execution.get("python", sys.executable)
        if "/" in python and not Path(python).is_absolute():
            python = str((base / python).resolve())
        rows.append(
            dict(
                stage=name,
                module=module,
                input=str(previous),
                input_kind=kind,
                output=str(target),
                output_kind=produces,
                options=options,
                python=python,
                timeout_seconds=timeout,
            )
        )
        previous, kind = target, produces
    return rows


def execute(config_path, *, selected=None, resume=False):
    config, base = read_config(config_path)
    rows = plan(config, base, selected=selected)
    output = (base / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        inp = Path(row["input"])
        target = Path(row["output"])
        if row["input_kind"] == "video":
            if not inp.is_file():
                raise FileNotFoundError(inp)
            input_hash = digest(inp)
        else:
            load(inp, kinds={row["input_kind"]})
            input_hash = digest(inp / "artifact.json")
        request = dict(row, config_base=str(base), input_sha256=input_hash)
        receipt = target / "execution.json"
        if receipt.exists() and resume:
            saved = json.loads(receipt.read_text())
            if saved["request"] != request:
                raise ValueError("Resume inputs/configuration differ: " + row["stage"])
            if saved["artifact_sha256"] != digest(target / "artifact.json"):
                raise ValueError("Completed artifact manifest changed: " + row["stage"])
            load(target, kinds={row["output_kind"]})
            continue
        if target.exists():
            raise FileExistsError(
                "Use a fresh output or --resume with a valid receipt: " + str(target)
            )
        target.mkdir()
        request_path = target / "request.json"
        write_json(request_path, request)
        command = [row["python"], "-m", row["module"], "--request", str(request_path)]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if row["options"].get("execution", {}).get("cpu_only", False):
            env["CUDA_VISIBLE_DEVICES"] = ""
        # GPU stages are supervised as process groups, including their children.
        with (target / "console.log").open("x") as log:
            from .supervisor import run

            run(
                command,
                cwd=base,
                env=env,
                stream=log,
                timeout=row["timeout_seconds"],
                thermal=row["options"].get("execution", {}).get("thermal"),
            )
        artifact = load(target, kinds={row["output_kind"]})
        write_json(
            receipt,
            dict(request=request, artifact_sha256=digest(target / "artifact.json")),
        )
        write_json(
            output / "progress.json",
            dict(completed_stage=row["stage"], artifact_kind=artifact["kind"]),
        )
    return Path(rows[-1]["output"])
