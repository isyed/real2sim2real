"""Thermally supervise one owned Docker container, not merely its CLI process.

Local execution requires a live
1500 MHz temporary-clock helper. Cloud telemetry may be supplied explicitly. This module never changes clocks, power, CPU
limits or privileges. The caller owns the shared lease's completion marker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid


def telemetry(_device="gpu"):
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=temperature.gpu,memory.used,clocks.gr",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    temperature, memory, clock = map(float, result.strip().split(","))
    sample = dict(temperature_c=temperature, memory_mib=memory, gpu_clock_mhz=clock)
    if not all(math.isfinite(x) for x in sample.values()):
        raise RuntimeError("Nonfinite GPU telemetry")
    return sample


OWNER_LABEL = "real2sim2real.thermal_guard"
PAUSE_C, RESUME_C, STOP_C = 82.0, 74.0, 83.0
GUARD_EXIT = 86


class GuardFailure(RuntimeError):
    pass


def thermal_action(sample: Mapping[str, Any], *, mode: str, paused: bool) -> str:
    """Pure fail-closed thermal/cap decision, including hysteresis."""
    required = [
        "temperature_c",
        "power_limit_w" if mode == "cloud" else "gpu_clock_mhz",
    ]
    if mode not in ("cloud", "local"):
        raise GuardFailure("Unknown guard mode")
    if any(
        key not in sample or not math.isfinite(float(sample[key])) for key in required
    ):
        raise GuardFailure("Required GPU telemetry unavailable or nonfinite")
    if mode == "cloud" and abs(float(sample["power_limit_w"]) - 60.0) > 0.1:
        raise GuardFailure("Cloud GPU must report the applied 60 W power limit")
    if mode == "local" and float(sample["gpu_clock_mhz"]) > 1500.0:
        raise GuardFailure("Local GPU exceeds the 1500 MHz clock ceiling")
    temperature = float(sample["temperature_c"])
    if temperature < 0:
        raise GuardFailure("Invalid negative GPU temperature")
    if temperature >= STOP_C:
        raise GuardFailure("Emergency thermal stop at 83 C")
    if not paused and temperature >= PAUSE_C:
        return "pause"
    if paused and temperature <= RESUME_C:
        return "unpause"
    return "hold"


def live_clock_helper(session: Path) -> bool:
    """Look for the actual Python helper, not a stale lease file or sudo parent."""
    wanted = session.resolve()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [
                item.decode()
                for item in (process / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if len(args) < 2 or not Path(args[0]).name.startswith("python"):
                continue
            if not any(
                Path(item).name == "temporary_gpu_clock.py" for item in args[1:]
            ):
                continue
            index = args.index("--session-dir")
            path = Path(args[index + 1])
            if not path.is_absolute():
                path = (process / "cwd").resolve() / path
            if path.resolve() == wanted:
                return True
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return False


def check_local_lease(
    session: Path,
    *,
    now_epoch: float | None = None,
    helper_alive: Callable[[Path], bool] = live_clock_helper,
) -> dict[str, Any]:
    if (session / "clock_session.done").exists():
        raise GuardFailure("The local clock lease is already completed")
    record = json.loads((session / "clock_session.json").read_text())
    if (
        record.get("status") != "active"
        or record.get("requested_max_clock_mhz") != 1500
    ):
        raise GuardFailure("An active 1500 MHz local clock lease is required")
    now = time.time() if now_epoch is None else now_epoch
    start, duration = float(record["started_epoch"]), float(record["max_seconds"])
    if (
        not all(math.isfinite(item) for item in (now, start, duration))
        or not 0 < duration <= 28800
    ):
        raise GuardFailure("Invalid local clock lease duration")
    if start > now + 5 or now >= start + duration:
        raise GuardFailure("The local clock lease is stale, expired or future-dated")
    if not helper_alive(session):
        raise GuardFailure(
            "The temporary GPU clock helper is not running for this lease"
        )
    return record


def validate_docker_command(command: Sequence[str], container_name: str) -> list[str]:
    """Validate Docker options before the image; image command arguments are inert.

    A deliberately small supported option set prevents confusing an image's own
    ``--name``/``--gpus`` arguments with Docker options. Unsupported options fail
    explicitly rather than making a different GPU/remote/detached job unguarded.
    """
    command = list(command)
    if len(command) < 3 or Path(command[0]).name != "docker" or command[1] != "run":
        raise ValueError("Expected a direct docker run command")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", container_name) is None:
        raise ValueError("Invalid container name")
    flags = {"--rm", "--init", "--read-only"}
    values = {
        "--name",
        "--gpus",
        "--network",
        "--cpus",
        "--memory",
        "--memory-swap",
        "--user",
        "--env",
        "-e",
        "--volume",
        "-v",
        "--mount",
        "--entrypoint",
        "--workdir",
        "-w",
        "--ipc",
        "--shm-size",
        "--runtime",
        "--security-opt",
        "--ulimit",
        "--tmpfs",
        "--pids-limit",
        "--platform",
        "--pull",
        "--label",
        "--hostname",
        "--add-host",
        "--group-add",
        "--cap-drop",
    }
    found: dict[str, list[str]] = {}
    index = 2
    while index < len(command) and command[index].startswith("-"):
        token = command[index]
        if token in flags:
            index += 1
            continue
        name, separator, value = token.partition("=")
        if name not in values:
            raise ValueError(f"Unsupported Docker option for guarded execution: {name}")
        if not separator:
            index += 1
            if index >= len(command):
                raise ValueError(f"Missing value for Docker option {name}")
            value = command[index]
        found.setdefault(name, []).append(value)
        index += 1
    if index >= len(command):
        raise ValueError("Docker image is missing")
    if found.get("--name") != [container_name]:
        raise ValueError("Docker must specify exactly the guarded container name")
    if found.get("--gpus") != ["device=0"]:
        raise ValueError("Docker must select only --gpus device=0, matching telemetry")
    if any(value.split("=", 1)[0] == OWNER_LABEL for value in found.get("--label", [])):
        raise ValueError("The thermal guard ownership label is reserved")
    if "--rm" not in command[2:index]:
        raise ValueError("Guarded Docker runs require --rm")
    return command


def _not_found(result: subprocess.CompletedProcess) -> bool:
    return bool(
        result.returncode
        and any(
            text in (result.stderr or "").lower()
            for text in ("no such object", "no such container")
        )
    )


class OwnedContainer:
    """All mutations target the immutable ID from our private Docker cidfile."""

    def __init__(
        self, docker: str, name: str, owner: str, cidfile: Path, run: Callable
    ):
        self.docker, self.name, self.owner, self.cidfile, self.run = (
            docker,
            name,
            owner,
            cidfile,
            run,
        )
        self.container_id: str | None = None

    def call(self, *args: str, timeout: int = 10):
        return self.run(
            [self.docker, "--context", "default", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def require_unused_name(self):
        result = self.call("container", "inspect", "--format", "{{.Id}}", self.name)
        if result.returncode == 0:
            raise GuardFailure(f"Container name already exists: {self.name}")
        if not _not_found(result):
            raise GuardFailure(
                f"Cannot verify Docker container name: {result.stderr.strip()}"
            )

    def require_local_daemon(self):
        result = self.call(
            "context", "inspect", "default", "--format", "{{.Endpoints.docker.Host}}"
        )
        if result.returncode or not result.stdout.strip().startswith("unix:///"):
            raise GuardFailure(
                "The Docker default context must use a local Unix socket"
            )

    def identity(self) -> str | None:
        if self.container_id is None and self.cidfile.exists():
            value = self.cidfile.read_text().strip()
            if not value:
                return None
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise GuardFailure("Docker published an invalid container ID")
            self.container_id = value
        return self.container_id

    def state(self) -> dict[str, bool] | None:
        cid = self.identity()
        if cid is None:
            return None
        template = (
            '{{index .Config.Labels "'
            + OWNER_LABEL
            + '"}}|{{.State.Running}}|{{.State.Paused}}'
        )
        result = self.call("container", "inspect", "--format", template, cid)
        if _not_found(result):
            return None
        if result.returncode:
            raise GuardFailure(f"Container inspection failed: {result.stderr.strip()}")
        fields = result.stdout.strip().split("|")
        if (
            len(fields) != 3
            or fields[0] != self.owner
            or any(item not in ("true", "false") for item in fields[1:])
        ):
            raise GuardFailure("Container ownership/state verification failed")
        return {"running": fields[1] == "true", "paused": fields[2] == "true"}

    def action(self, action: str):
        cid = self.identity()
        if cid is None:
            raise GuardFailure("Cannot control a container without its owned ID")
        result = self.call(action, cid)
        if result.returncode and not _not_found(result):
            raise GuardFailure(f"docker {action} failed: {result.stderr.strip()}")

    def cleanup(self, *, emergency: bool):
        cid = self.identity()
        if cid is None:
            return
        try:
            state = self.state()
        except Exception:
            # The private cidfile proves ownership even if inspect failed.
            state = {"running": True, "paused": True}
        if state is None:
            return
        if not state["running"]:
            result = self.call("rm", cid)
            if result.returncode and not _not_found(result):
                raise GuardFailure(
                    f"Unable to remove unstarted owned container {cid}: {result.stderr.strip()}"
                )
            return
        if state["paused"]:
            try:
                self.action("unpause")
            except Exception:
                pass
        result = None
        try:
            result = (
                self.call("kill", cid)
                if emergency
                else self.call("stop", "--time", "5", cid)
            )
        except Exception:
            pass
        if result is None or (result.returncode and not _not_found(result)):
            result = self.call("kill", cid)
            if result.returncode and not _not_found(result):
                raise GuardFailure(
                    f"Unable to stop owned container {cid}: {result.stderr.strip()}"
                )


def guard_container(
    command: Sequence[str],
    *,
    container_name: str,
    mode: str,
    session: Path | None,
    max_seconds: float,
    log: Path,
    run: Callable = subprocess.run,
    popen: Callable = subprocess.Popen,
    read_sample: Callable[[], Mapping[str, Any]] = lambda: telemetry("gpu"),
    lease_check: Callable[[Path], Any] = check_local_lease,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    command = validate_docker_command(command, container_name)
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError("max_seconds must be positive and at most four hours")
    if mode not in ("local", "cloud") or (mode == "local" and session is None):
        raise ValueError(
            "Local/cloud mode required; local mode also requires --session"
        )
    # Explicit context pins the local default daemon, overriding remote current
    # contexts and DOCKER_HOST. Never supervise a remote GPU with local sensors.
    owner = uuid.uuid4().hex
    log.parent.mkdir(parents=True, exist_ok=True)
    with (
        log.open("a", buffering=1) as stream,
        tempfile.TemporaryDirectory(prefix="r2s2r-guard-") as temporary,
    ):

        def emit(event: str, **values):
            stream.write(
                json.dumps(
                    {
                        "epoch": time.time(),
                        "event": event,
                        "container_name": container_name,
                        **values,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

        container = OwnedContainer(
            command[0], container_name, owner, Path(temporary) / "container.id", run
        )
        child = None
        failure = False
        started = monotonic()
        try:
            container.require_local_daemon()
            container.require_unused_name()
            if mode == "local":
                lease_check(session)
            sample = dict(read_sample())
            if thermal_action(sample, mode=mode, paused=False) == "pause":
                raise GuardFailure("GPU must be below 82 C before starting")
            emit("preflight_passed", mode=mode, max_seconds=max_seconds, sample=sample)
            # Create first: there can be no untracked GPU work if the client is
            # interrupted before Docker publishes the container's immutable ID.
            owned_command = [
                command[0],
                "--context",
                "default",
                "create",
                "--cidfile",
                str(container.cidfile),
                "--label",
                f"{OWNER_LABEL}={owner}",
                *command[2:],
            ]
            created = run(
                owned_command,
                capture_output=True,
                text=True,
                timeout=min(60.0, max_seconds),
            )
            if created.returncode:
                raise GuardFailure(f"Docker create failed: {created.stderr.strip()}")
            state = container.state()
            if state is None or state["running"]:
                raise GuardFailure("Expected an owned, created but unstarted container")
            if monotonic() - started >= max_seconds:
                raise GuardFailure("Container runtime deadline reached before start")
            if mode == "local":
                lease_check(session)
            sample = dict(read_sample())
            if thermal_action(sample, mode=mode, paused=False) == "pause":
                raise GuardFailure("GPU must be below 82 C before container start")
            emit(
                "container_created", container_id=container.container_id, sample=sample
            )
            child = popen(
                [
                    command[0],
                    "--context",
                    "default",
                    "start",
                    "--attach",
                    container.container_id,
                ],
                start_new_session=True,
            )
            while child.poll() is None:
                elapsed = monotonic() - started
                if elapsed >= max_seconds:
                    raise GuardFailure("Container runtime deadline reached")
                if mode == "local":
                    lease_check(session)
                sample = dict(read_sample())
                state = container.state()
                action = thermal_action(
                    sample, mode=mode, paused=bool(state and state["paused"])
                )
                if action == "pause" and state is None:
                    raise GuardFailure(
                        "GPU reached pause threshold before container became controllable"
                    )
                if state and state["running"] and action != "hold":
                    container.action(action)
                    emit(action, elapsed_s=elapsed, sample=sample)
                else:
                    emit(
                        "sample",
                        elapsed_s=elapsed,
                        paused=bool(state and state["paused"]),
                        sample=sample,
                    )
                sleep(0.5)
            emit("process_exit", returncode=child.returncode)
            result = int(child.returncode)
        except (Exception, KeyboardInterrupt) as exc:
            failure = True
            result = GUARD_EXIT
            emit("guard_stop", error=f"{type(exc).__name__}: {exc}")
            print(f"GPU GUARD STOP: {type(exc).__name__}: {exc}", flush=True)
        finally:
            # The start client is created only after identity verification. This
            # branch also handles an interrupted/abnormal client implementation.
            if (
                child is not None
                and child.poll() is None
                and container.identity() is None
            ):
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            try:
                container.cleanup(emergency=failure)
            except Exception as exc:
                result = GUARD_EXIT
                emit("cleanup_failed", error=f"{type(exc).__name__}: {exc}")
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            emit("finished", returncode=result, container_id=container.container_id)
        return result


@contextmanager
def signal_cleanup():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, _frame):
        # Ignore repeated interruption while the finally block stops Docker.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Signal {signum}")

    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--mode", choices=("local", "cloud"), required=True)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    with signal_cleanup():
        return guard_container(
            command,
            container_name=args.container_name,
            mode=args.mode,
            session=args.session,
            max_seconds=args.max_seconds,
            log=args.log,
        )


if __name__ == "__main__":
    raise SystemExit(main())
