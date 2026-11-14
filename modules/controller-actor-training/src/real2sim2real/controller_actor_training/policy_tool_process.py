"""JSON-only calls to one persistent, separately hosted policy tool service."""

import json
import select
import subprocess
import threading


class PolicyToolProcess:
    def __init__(self, command, *, env, cwd, log, timeout_s=280):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
            cwd=cwd,
        )
        self.timeout = timeout_s
        self.lock = threading.Lock()

    def call(self, tool, **inputs):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("Physics service has ended")
            self.process.stdin.write(
                json.dumps({"tool": tool, "inputs": inputs}, allow_nan=False) + "\n"
            )
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
            if not ready:
                self.process.kill()
                raise TimeoutError("Physics tool deadline exceeded")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Physics service returned no reply")
            reply = json.loads(line)
            if not reply["ok"]:
                raise ValueError(reply["error"])
            return reply["value"]

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise


class RemotePolicyTools:
    """Typed GaP-facing proxy; only JSON travels to the persistent simulator."""

    def __init__(self, process):
        self.process = process

    def describe(self) -> dict:
        return self.process.call("describe")

    def observe(self) -> dict:
        return self.process.call("observe")

    def plan_wrist(self, pose_world: list, seed_joints_rad: dict) -> dict:
        return self.process.call(
            "plan_wrist", pose_world=pose_world, seed_joints_rad=seed_joints_rad
        )

    def fit_hand(
        self,
        seed_joints_rad: dict,
        pad_indices: list,
        targets_world_m: list,
        inward_normals_world: list,
    ) -> dict:
        return self.process.call(
            "fit_hand",
            seed_joints_rad=seed_joints_rad,
            pad_indices=pad_indices,
            targets_world_m=targets_world_m,
            inward_normals_world=inward_normals_world,
        )

    def screen_path(self, positions: list, durations_s: list) -> dict:
        return self.process.call(
            "screen_path", positions=positions, durations_s=durations_s
        )

    def execute_joint_path(self, positions: list, durations_s: list) -> dict:
        return self.process.call(
            "execute_joint_path", positions=positions, durations_s=durations_s
        )

    def hold(self, duration_s: float) -> dict:
        return self.process.call("hold", duration_s=duration_s)

    def set_pressure(self, targets_n: list) -> dict:
        return self.process.call("set_pressure", targets_n=targets_n)

    def release(self, open_joints_rad: dict, duration_s: float) -> dict:
        return self.process.call(
            "release", open_joints_rad=open_joints_rad, duration_s=duration_s
        )
