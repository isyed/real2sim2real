"""Dispatch reconstruction phases into explicit Python/container environments.

Only standard-library dependencies run here; model dependencies stay in workers.
"""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
from real2sim2real.artifacts import digest, write_json
from real2sim2real.stage import request_main

PHASES = (
    "video",
    "segment",
    "pack_masks",
    "depth",
    "export_human",
    "human",
    "depth_alignment",
    "metric_inputs",
    "metric_mesh",
    "object_export",
    "object_motion",
    "publish",
)
GPU_PHASES = frozenset(
    {"segment", "depth", "human", "depth_alignment", "metric_mesh", "object_motion"}
)


def command(request, phase, request_path):
    if phase not in PHASES:
        raise ValueError("Unknown reconstruction phase: " + phase)
    options = request["options"]
    prefix = options.get("backend_commands", {}).get(phase)
    if prefix is None:
        interpreter = options.get("backend_pythons", {}).get(phase, sys.executable)
        if "/" in interpreter and not Path(interpreter).is_absolute():
            interpreter = str((Path(request["config_base"]) / interpreter).resolve())
        prefix = [interpreter]
    if (
        not isinstance(prefix, list)
        or not prefix
        or any(not isinstance(x, str) or not x or "\x00" in x for x in prefix)
    ):
        raise ValueError("Backend commands must be nonempty argument lists")
    return [
        *prefix,
        "-m",
        "real2sim2real.reconstruction.backend",
        "--phase",
        phase,
        "--request",
        str(request_path),
    ]


def run(request):
    output = Path(request["output"])
    if digest(Path(request["input"])) != request["input_sha256"]:
        raise ValueError("Source video changed before reconstruction")
    output.mkdir(parents=True, exist_ok=True)
    phase_request = output / "backend_request.json"
    if phase_request.exists():
        raise FileExistsError("Use a fresh output for reconstruction: " + str(output))
    timeout = request["options"].get("backend_timeout_seconds", 7200)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or timeout <= 0
    ):
        raise ValueError("Backend timeout must be positive")
    # Validate all commands before the first model process.
    commands = [
        dict(phase=p, command=command(request, p, phase_request)) for p in PHASES
    ]
    write_json(phase_request, request)
    write_json(output / "commands.json", commands)
    for item in commands:
        phase, argv = item["phase"], item["command"]
        started = time.time()
        write_json(
            output / "phase_status.json",
            dict(phase=phase, state="running", started=started),
        )
        try:
            with (output / (phase + ".log")).open("x") as log:
                subprocess.run(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
        except BaseException as error:
            write_json(
                output / "phase_status.json",
                dict(
                    phase=phase,
                    state="failed",
                    started=started,
                    finished=time.time(),
                    error=type(error).__name__,
                ),
            )
            raise
        write_json(
            output / "phase_status.json",
            dict(phase=phase, state="complete", started=started, finished=time.time()),
        )


if __name__ == "__main__":
    request_main(run)
