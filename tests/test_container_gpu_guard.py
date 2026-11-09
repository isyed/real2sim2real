import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from real2sim2real.runtime.container_guard import (
    GUARD_EXIT,
    OWNER_LABEL,
    GuardFailure,
    check_local_lease,
    guard_container,
    thermal_action,
    validate_docker_command,
)


def command(*extra):
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        "guard-test",
        "--gpus",
        "device=0",
        *extra,
        "image",
        "python",
        "task.py",
    ]


def sample(temp=40, power=60, clock=1000):
    return {"temperature_c": temp, "power_limit_w": power, "gpu_clock_mhz": clock}


def test_hysteresis_and_emergency():
    assert thermal_action(sample(81.9), mode="cloud", paused=False) == "hold"
    assert thermal_action(sample(82), mode="cloud", paused=False) == "pause"
    assert thermal_action(sample(80), mode="cloud", paused=True) == "hold"
    assert thermal_action(sample(74), mode="cloud", paused=True) == "unpause"
    with pytest.raises(GuardFailure, match="Emergency"):
        thermal_action(sample(83), mode="cloud", paused=True)


@pytest.mark.parametrize(
    "mode,reading",
    [
        ("cloud", sample(power=72)),
        ("local", sample(clock=1501)),
        ("cloud", sample(float("nan"))),
        ("local", {"temperature_c": 40}),
    ],
)
def test_caps_and_sensor_failure(mode, reading):
    with pytest.raises(GuardFailure):
        thermal_action(reading, mode=mode, paused=False)


def test_command_scope_and_image_arguments():
    assert validate_docker_command(command("--env", "VALUE=--name"), "guard-test")
    valid = command() + ["--name", "another", "--gpus", "all"]
    assert validate_docker_command(valid, "guard-test") == valid
    for invalid in [
        command("-d"),
        command("--privileged"),
        command("--name", "another"),
        command("--gpus", "all"),
        command("--label", f"{OWNER_LABEL}=mine"),
        command("--restart", "always"),
        command("--cidfile", "/tmp/untrusted"),
    ]:
        with pytest.raises(ValueError):
            validate_docker_command(invalid, "guard-test")


def test_local_lease_requires_active_helper_and_unexpired_record(tmp_path):
    record = {
        "status": "active",
        "requested_max_clock_mhz": 1500,
        "started_epoch": 1000,
        "max_seconds": 100,
    }
    (tmp_path / "clock_session.json").write_text(json.dumps(record))
    assert check_local_lease(tmp_path, now_epoch=1050, helper_alive=lambda _: True)
    with pytest.raises(GuardFailure, match="not running"):
        check_local_lease(tmp_path, now_epoch=1050, helper_alive=lambda _: False)
    with pytest.raises(GuardFailure, match="expired"):
        check_local_lease(tmp_path, now_epoch=1100, helper_alive=lambda _: True)
    (tmp_path / "clock_session.done").touch()
    with pytest.raises(GuardFailure, match="completed"):
        check_local_lease(tmp_path, now_epoch=1050, helper_alive=lambda _: True)


class FakeDocker:
    cid = "a" * 64

    def __init__(self, temperatures, *, existing=False, inspect_failure=False):
        self.temperatures = iter(temperatures)
        self.calls = []
        self.running = False
        self.paused = False
        self.existing = existing
        self.owner = None
        self.exists = False
        self.now = 0.0
        self.child = None
        self.inspect_failure = inspect_failure

    def run(self, args, **_kwargs):
        self.calls.append(args)
        action = args[3]
        if action == "context":
            return SimpleNamespace(
                returncode=0, stdout="unix:///var/run/docker.sock", stderr=""
            )
        if action == "container":
            if args[-1] == "guard-test":
                return SimpleNamespace(
                    returncode=0 if self.existing else 1,
                    stdout=self.cid if self.existing else "",
                    stderr="No such object",
                )
            if self.inspect_failure:
                raise RuntimeError("Docker inspect failed")
            if not self.exists:
                return SimpleNamespace(returncode=1, stdout="", stderr="No such object")
            return SimpleNamespace(
                returncode=0,
                stderr="",
                stdout=f"{self.owner}|{str(self.running).lower()}|{str(self.paused).lower()}",
            )
        if action == "create":
            Path(args[args.index("--cidfile") + 1]).write_text(self.cid)
            self.owner = args[args.index("--label") + 1].split("=", 1)[1]
            self.exists = True
            return SimpleNamespace(returncode=0, stderr="", stdout=self.cid)
        assert args[-1] == self.cid, (
            "Every mutation must target the owned immutable container ID"
        )
        if action == "pause":
            self.paused = True
        elif action == "unpause":
            self.paused = False
        elif action in ("stop", "kill"):
            self.running = False
            self.exists = False
            if self.child:
                self.child.returncode = 137 if action == "kill" else 0
        elif action == "rm":
            self.exists = False
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def popen(self, args, **_kwargs):
        assert args[1:] == ["--context", "default", "start", "--attach", self.cid]
        self.running = True
        self.child = SimpleNamespace(returncode=None)
        self.child.poll = lambda: self.child.returncode
        self.child.terminate = lambda: setattr(self.child, "returncode", -15)
        self.child.kill = lambda: setattr(self.child, "returncode", -9)
        self.child.wait = lambda **_: self.child.returncode
        return self.child

    def read(self):
        value = next(self.temperatures)
        if isinstance(value, BaseException):
            raise value
        if value == "exit":
            self.child.returncode = 0
            self.running = False
            self.exists = False
            return sample()
        return sample(value)

    def sleep(self, interval):
        self.now += interval

    def execute(
        self, tmp_path, *, max_seconds=10, mode="cloud", lease_check=lambda _: None
    ):
        return guard_container(
            command(),
            container_name="guard-test",
            mode=mode,
            session=tmp_path,
            max_seconds=max_seconds,
            log=tmp_path / "guard.jsonl",
            run=self.run,
            popen=self.popen,
            read_sample=self.read,
            monotonic=lambda: self.now,
            sleep=self.sleep,
            lease_check=lease_check,
        )


def test_container_itself_pauses_and_resumes_then_exits(tmp_path):
    docker = FakeDocker([40, 40, 82, 80, 74, "exit"])
    assert docker.execute(tmp_path) == 0
    actions = [c[3] for c in docker.calls]
    assert actions.count("pause") == actions.count("unpause") == 1
    assert "kill" not in actions and "stop" not in actions


@pytest.mark.parametrize(
    "failure", [83, RuntimeError("sensor unavailable"), KeyboardInterrupt("SIGTERM")]
)
def test_emergency_sensor_and_interruption_cleanup_paused_container(tmp_path, failure):
    docker = FakeDocker([40, 40, 82, failure])
    assert docker.execute(tmp_path) == GUARD_EXIT
    actions = [c[3] for c in docker.calls]
    assert actions[-2:] == ["unpause", "kill"]
    assert not docker.running


def test_deadline_kills_container_and_no_existing_container_is_touched(tmp_path):
    docker = FakeDocker([40, 40, 40, 40])
    assert docker.execute(tmp_path, max_seconds=0.5) == GUARD_EXIT
    assert any(c[3] == "kill" for c in docker.calls)
    other = FakeDocker([40], existing=True)
    assert other.execute(tmp_path) == GUARD_EXIT
    assert other.child is None
    assert all(c[3] in ("context", "container") for c in other.calls)


def test_inspection_failure_still_kills_owned_cid(tmp_path):
    docker = FakeDocker([40, 40], inspect_failure=True)
    assert docker.execute(tmp_path) == GUARD_EXIT
    assert any(c[3] == "kill" and c[-1] == docker.cid for c in docker.calls)


def test_lease_expiry_during_execution_kills_container(tmp_path):
    docker = FakeDocker([40, 40, 40, 40])
    calls = []

    def lease(_):
        calls.append(1)
        if len(calls) >= 4:
            raise GuardFailure("Lease expired")

    assert docker.execute(tmp_path, mode="local", lease_check=lease) == GUARD_EXIT
    assert not docker.running
    assert not (tmp_path / "clock_session.done").exists(), (
        "Per-container guard must not close the shared lease"
    )


def test_hot_after_create_never_starts_gpu_and_removes_created_container(tmp_path):
    docker = FakeDocker([40, 82])
    assert docker.execute(tmp_path) == GUARD_EXIT
    assert docker.child is None
    assert any(c[3] == "rm" for c in docker.calls)
    assert not docker.exists


def test_detached_container_is_stopped_if_attach_client_exits(tmp_path):
    docker = FakeDocker([40, 40])
    real_popen = docker.popen

    def disconnect(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child.returncode = 1
        return child

    docker.popen = disconnect
    assert docker.execute(tmp_path) == 1
    assert any(c[3] == "stop" for c in docker.calls)
    assert not docker.running


def test_remote_docker_context_is_rejected_without_create(tmp_path):
    docker = FakeDocker([40])
    real_run = docker.run

    def remote(args, **kwargs):
        if args[3] == "context":
            docker.calls.append(args)
            return SimpleNamespace(
                returncode=0, stdout="ssh://remote-server", stderr=""
            )
        return real_run(args, **kwargs)

    docker.run = remote
    assert docker.execute(tmp_path) == GUARD_EXIT
    assert docker.child is None
    assert all(c[3] == "context" for c in docker.calls)


def test_stop_failure_falls_back_to_kill(tmp_path):
    docker = FakeDocker([40, 40])
    real_popen, real_run = docker.popen, docker.run

    def disconnect(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child.returncode = 1
        return child

    def stop_error(args, **kwargs):
        if args[3] == "stop":
            docker.calls.append(args)
            return SimpleNamespace(returncode=1, stdout="", stderr="stop failed")
        return real_run(args, **kwargs)

    docker.popen, docker.run = disconnect, stop_error
    assert docker.execute(tmp_path) == 1
    assert [c[3] for c in docker.calls][-2:] == ["stop", "kill"]
    assert not docker.running


def test_local_lease_accepts_requested_eight_hours_but_rejects_longer(tmp_path):
    record = {
        "status": "active",
        "requested_max_clock_mhz": 1500,
        "started_epoch": 1000,
        "max_seconds": 28800,
    }
    (tmp_path / "clock_session.json").write_text(json.dumps(record))
    assert check_local_lease(tmp_path, now_epoch=29000, helper_alive=lambda _: True)
    with pytest.raises(GuardFailure):
        check_local_lease(tmp_path, now_epoch=29800, helper_alive=lambda _: True)
    record["max_seconds"] = 28801
    (tmp_path / "clock_session.json").write_text(json.dumps(record))
    with pytest.raises(GuardFailure):
        check_local_lease(tmp_path, now_epoch=1001, helper_alive=lambda _: True)
