"""Local Docker adapter with explicit source/model mounts and owned GPU guard."""

from pathlib import Path
import os
import subprocess
import time
from real2sim2real.artifacts import digest, write_json


def docker_command(
    config, output, phase, *, check=False, cpu_only=False, request_path=None
):
    root = Path(config["source_root"]).resolve()
    out = Path(output).resolve()
    engine = root / "modules/reconstruction/src/r2s2r_engines"
    source_paths = [
        root / "src",
        root / "modules/reconstruction/src",
        root / "modules/egocentric-reconstruction/src",
        engine / "foundation_pose/lib/FoundationPose",
    ]
    source_paths.extend(
        Path(p) for p in config.get("native_python_paths", {}).get(phase, [])
    )
    name = "r2s2r-ego-" + phase + "-" + str(time.time_ns())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--name",
        name,
        "--pull=never",
        "--cpus",
        "4",
        "--memory",
        "40g",
        "--memory-swap",
        "40g",
        "--shm-size",
        "8g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    cmd += (
        [
            "--runtime",
            "runc",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
        ]
        if check or cpu_only
        else ["--gpus", "device=0"]
    )
    env = dict(
        PYTHONPATH=os.pathsep.join(map(str, source_paths)),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        XDG_CACHE_HOME="/tmp/cache",
        MPLCONFIGDIR="/tmp/matplotlib",
        PYOPENGL_PLATFORM="egl",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        R2S2R_FP_RENDER_BATCH="16",
        R2S2R_FP_REFINE_BATCH="32",
    )
    weights = config["weights"]
    if phase == "mesh":
        env.update(
            HF_HOME=str(Path(weights["sam3d"]) / "hf_home"),
            TORCH_HOME=str(Path(weights["sam3d"]) / "torch_home"),
            LIDRA_SKIP_INIT="1",
        )
    for key, value in env.items():
        cmd += ["-e", key + "=" + value]
    # Only source, model stores, and this run's predictions are visible to inference.
    for path, ro in [
        (root, True),
        (out, False),
        *[(Path(v).resolve(), True) for v in weights.values()],
        *[(Path(v).resolve(), True) for v in config.get("read_only_inputs", [])],
    ]:
        cmd += ["-v", str(path) + ":" + str(path) + (":ro" if ro else "")]
    cmd += [
        "--entrypoint",
        "python",
        config["images"][phase],
        "-m",
        "real2sim2real.egocentric.worker",
        "--request",
        str(request_path or out / "worker_request.json"),
        "--phase",
        phase,
    ]
    if check:
        cmd += ["--check"]
    return cmd, name


def run_model(config, output, phase, *, check=False):
    out = Path(output).resolve()
    result = {
        "segment": "masks",
        "mesh": "mesh/metric.glb",
        "track": "object_motion.npz",
        "hands": "hands_raw/predictions.npz",
        "refine-hands": "hands_metric/hands.npz",
        "temporal-hands": "hands_metric/hands.npz",
    }[phase]
    if not check and (out / result).exists():
        raise FileExistsError(
            f"Retain existing {phase} results; select a new output directory for another hypothesis"
        )
    attempt = out / "attempts" / phase / str(time.time_ns())
    attempt.mkdir(parents=True)
    request = dict(
        output=str(out), weights=config["weights"], **config.get("model_options", {})
    )
    write_json(attempt / "request.json", request)
    cpu_only = phase in ("refine-hands", "temporal-hands")
    cmd, name = docker_command(
        config,
        out,
        phase,
        check=check,
        cpu_only=cpu_only,
        request_path=attempt / "request.json",
    )
    log = attempt / "worker.log"
    root = Path(config["source_root"]).resolve()
    files = {
        str(p.relative_to(root)): digest(p)
        for folder in [
            "src",
            "modules/egocentric-reconstruction/src",
            "modules/reconstruction/src",
        ]
        for p in (root / folder).rglob("*.py")
    }
    write_json(
        attempt / "source.json",
        dict(
            command=cmd,
            files=files,
            request=request,
            check_only=check,
            cpu_only=check or cpu_only,
        ),
    )
    write_json(
        out / (phase + (".check" if check else "") + ".latest.json"),
        dict(attempt=str(attempt), check_only=check),
    )
    if check:
        with log.open("w") as stream:
            subprocess.run(
                cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180
            )
        return
    if cpu_only:
        timeout = config.get("timeouts", {}).get(phase, 2400)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 28800
        ):
            raise ValueError(
                "CPU refinement requires a finite deadline of at most eight hours"
            )
        try:
            with log.open("w") as stream:
                subprocess.run(
                    cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
        finally:
            # The unique name belongs to this invocation; also clean up on timeout.
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
        return
    # A separate host process owns stdout so container output is both persistent
    # and visible. The guard itself owns container cleanup by immutable ID.
    guard = [
        "python",
        "-m",
        "real2sim2real.runtime.container_guard",
        "--container-name",
        name,
        "--mode",
        "local",
        "--session",
        config["gpu_session"],
        "--log",
        str(attempt / "thermal.jsonl"),
        "--max-seconds",
        str(config.get("timeouts", {}).get(phase, 3600)),
        "--",
        *cmd,
    ]
    import sys

    guard[0] = sys.executable
    with log.open("w") as stream:
        from real2sim2real.bounded_process import run

        run(
            guard,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=config.get("timeouts", {}).get(phase, 3600) + 90,
        )
