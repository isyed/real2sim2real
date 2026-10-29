"""Bound native subprocess groups in time and optionally by GPU temperature."""

import os
import signal
import subprocess
import time


def run(command, *, cwd, env, stream, timeout, thermal=None):
    if thermal:
        pause, resume, stop = [
            float(thermal[k]) for k in ("pause_c", "resume_c", "stop_c")
        ]
        if not 0 < resume < pause < stop < 100:
            raise ValueError(
                "Thermal limits must satisfy 0 < resume < pause < stop < 100"
            )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    paused = False
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Stage time limit reached")
            if thermal:
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(thermal.get("gpu", 0)),
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                temperature = float(sample.stdout.strip())
                if temperature >= stop:
                    raise RuntimeError("GPU emergency temperature limit reached")
                if temperature >= pause and not paused:
                    os.killpg(process.pid, signal.SIGSTOP)
                    paused = True
                elif temperature <= resume and paused:
                    os.killpg(process.pid, signal.SIGCONT)
                    paused = False
            time.sleep(1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            if paused:
                os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
