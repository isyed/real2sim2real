"""Reset/step CPU environment for bounded pressure residuals on frozen arm control.

No learning or GPU work is started by this module. Observations are privileged
simulator estimates. This scalar NumPy interface is not the vectorized G1 backend.
"""

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .residual_pressure import CHANNELS, PressureResidual, validate_action
from .arm_residual_observation import ArmResidualObservation, transition_reward


def frozen_source(config):
    required = {
        "baseline_result",
        "baseline_sha256",
        "output",
        "residual",
        "deadline_s",
    }
    if set(config) != required or not 0 < config["deadline_s"] <= 900:
        raise ValueError(
            "Pinned baseline, output, pressure settings and <=900 s deadline required"
        )
    path = Path(config["baseline_result"])
    if digest(path) != config["baseline_sha256"]:
        raise ValueError("Frozen controller result changed")
    result = json.loads(path.read_text())
    if (
        result.get("schema") != "real2sim2real.arm-replay-pilot/1"
        or result.get("residual_adapter") is not None
        or result.get("bounded_grasp_correction") is not None
        or result.get("closed_loop_grasp_feedback") is not None
        or result.get("finger_control_mode") != "live_contact_force"
        or result.get("physics_run") is not True
        or "incomplete_pilot" in result["rejected_for"]
        or not result.get("fixed_controller_clock")
    ):
        raise ValueError(
            "Complete fixed-clock live-finger frozen replay required; residual stacking is not supported"
        )
    if result.get("continuous_protocol") is not None:
        from .continuous_residual import validate_baseline

        validate_baseline(result)
    for filename, field in (
        ("scene.xml", "scene_sha256"),
        ("states.npz", "states_sha256"),
    ):
        if digest(path.with_name(filename)) != result[field]:
            raise ValueError("Frozen physical artifacts changed")
    return result


class PressureBridge(PressureResidual):
    """Internal coroutine bridge; policy callers receive copied arrays only."""

    def __init__(self, settings, deadline_s):
        super().__init__(settings)
        self.deadline = time.monotonic() + deadline_s
        self.packets, self.actions = [], []
        self.implementation = {
            name: Path(__file__).with_name(name).read_bytes()
            for name in (
                "arm_residual_env.py",
                "arm_residual_observation.py",
                "residual_pressure.py",
                "continuous_residual.py",
            )
        }

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Residual episode wall-clock deadline exceeded")

    def bind(self, model, data, fingers, roots, parent, times, control_period):
        ratio = self.settings["policy_period_s"] / control_period
        if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0) or ratio < 1:
            raise ValueError(
                "Policy period must be an integer controller-period multiple"
            )
        self.policy_stride = int(round(ratio))
        self.observer = ArmResidualObservation(
            model, data, fingers, roots, parent, times
        )
        self.horizon_s = float(times[-1] - times[0])

    def record(self, record):
        self.observer.record(record)

    def observe(self, time_s, i, alpha):
        packet = self.observer.capture(time_s, i, alpha, self)
        self.packets.append(packet)
        return packet

    def accept(self, action):
        super().accept(action)
        self.actions.append(self.requested.copy())

    def save_trace(self, output):
        np.savez_compressed(
            output / "policy-trace.npz",
            observations=[p["observation"] for p in self.packets],
            policy_times_s=[p["time_s"] for p in self.packets],
            actions=self.actions,
        )
        return digest(output / "policy-trace.npz")

    def finish(self, report, output):
        now = float(self.observer.live.time)
        times = self.observer.times
        i = int(
            np.clip(np.searchsorted(times, now, side="right") - 1, 0, len(times) - 2)
        )
        alpha = float(np.clip((now - times[i]) / (times[i + 1] - times[i]), 0, 1))
        try:
            self.observe(now, i, alpha)
        except ValueError:
            if "incomplete_pilot" not in report["rejected_for"]:
                raise
            packet = deepcopy(self.packets[-1])
            packet["time_s"] = now
            packet["terminal_observation_valid"] = False
            self.packets.append(packet)
        trace_sha = self.save_trace(output)
        sources = {}
        for name, contents in self.implementation.items():
            target = output / "implementation" / name
            target.write_bytes(contents)
            sources[name] = digest(target)
        return dict(
            schema="real2sim2real.arm-pressure-residual/1",
            settings=self.settings,
            action_channels=list(CHANNELS),
            policy_actions=len(self.actions),
            all_actions_zero=bool(not np.any(self.actions)),
            peak_requested_pressure_offset_n=self.peak.tolist(),
            pressure_target_limits_n=[0.1, 10.0],
            observation_source="privileged separate-data simulator forward solve; previous commands and filter state",
            observation_schema=self.observer.schema,
            phase_order=self.observer.phases,
            task_stage_order=getattr(self.observer, "stages", []),
            release_owner="frozen table-supported finger release; pressure actor cannot override opening",
            joint_order=[
                self.observer.model.joint(j).name for j in self.observer.joints
            ],
            actuator_order=[
                self.observer.model.actuator(a).name
                for a in range(self.observer.model.nu)
            ],
            pad_order=[
                p["role"] + ":" + p["group"] for p in self.observer.fingers.servo.pads
            ],
            trace_sha256=trace_sha,
            implementation_sha256=sources,
            hardware_ready=False,
        )


class ArmResidualEnv:
    """Gym-style scalar reset/step API, without mandatory Gym/JAX dependencies.

    reset -> (observation, info)
    step -> (observation, reward, terminated, truncated, info)

    reset creates fresh physics and controller/filter state each time. Complete
    failed trials terminate normally with success=False. Infrastructure timeouts
    truncate and must be discarded by a learner. No implicit auto-reset occurs.
    """

    action_dim = 4

    def __init__(self, config):
        self.config = deepcopy(config)
        self.baseline = frozen_source(self.config)
        PressureResidual(self.config["residual"])
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=False)
        self.frame_dt = config["residual"]["policy_period_s"]
        self.episode = -1
        self.worker = None
        self.finished = True
        self.observation_dim = None
        self.report = None

    def reset(self, *, seed=None, options=None):
        if options or (
            seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))
        ):
            raise ValueError(
                "No task overrides; optional integer seed only (deterministic environment)"
            )
        self.close()
        self.baseline = frozen_source(self.config)
        self.episode += 1
        self.seed = seed
        self.report = None
        self.rewards = []
        self.directory = self.output / f"episode-{self.episode:04d}"
        replay = deepcopy(self.baseline["config"])
        replay["output"] = str(self.directory)
        self.bridge = PressureBridge(self.config["residual"], self.config["deadline_s"])
        from .arm_replay_pilot import rollout

        self.worker = rollout(replay, interaction=self.bridge)
        try:
            self.packet = next(self.worker)
        except BaseException:
            self.worker.close()
            self.worker = None
            raise
        self.finished = False
        self.observation_dim = len(self.packet["observation"])
        return self.packet["observation"].copy(), self.info()

    def info(self):
        return dict(
            phase=self.packet["phase"],
            task_stage=self.packet.get("task_stage"),
            time_s=self.packet["time_s"],
            observation_schema=deepcopy(self.bridge.observer.schema),
            privileged_observations=True,
            baseline_success=self.baseline["success"],
            baseline_rejected_for=list(self.baseline["rejected_for"]),
            learning_costs=dict(self.packet["costs"]),
            action_channels=list(CHANNELS),
            terminal_observation_valid=self.packet.get(
                "terminal_observation_valid", True
            ),
        )

    def step(self, action):
        if self.worker is None or self.finished:
            raise RuntimeError("Reset before stepping a new episode")
        action = validate_action(
            action
        )  # Reject invalid actions before advancing physics.
        before = self.packet
        terminated = truncated = False
        try:
            self.bridge.check_deadline()
            self.packet = self.worker.send(action)
        except StopIteration as done:
            self.report = done.value
            self.packet = self.bridge.packets[-1]
            self.finished = terminated = True
        except TimeoutError:
            self.worker.close()
            self.finished = truncated = True
            self.rewards.append(-110.0)
            write_json(
                self.directory / "interrupted.json",
                dict(
                    reason="wall_clock_deadline", success=False, training_eligible=False
                ),
            )
            return (
                before["observation"].copy(),
                -110.0,
                False,
                True,
                dict(
                    self.info(), training_eligible=False, reason="wall_clock_deadline"
                ),
            )
        except Exception as error:
            self.worker.close()
            self.finished = True
            write_json(
                self.directory / "interrupted.json",
                dict(
                    reason=type(error).__name__,
                    success=False,
                    training_eligible=False,
                ),
            )
            raise
        reward = transition_reward(
            before,
            self.packet,
            horizon_s=self.bridge.horizon_s,
            success=self.report["success"] if terminated else None,
            failure=terminated and "incomplete_pilot" in self.report["rejected_for"],
        )
        self.rewards.append(reward)
        info = self.info()
        if terminated:
            info.update(
                success=self.report["success"],
                rejected_for=list(self.report["rejected_for"]),
                report=str(self.directory / "result.json"),
            )
            write_json(
                self.directory / "environment.json",
                dict(
                    schema="real2sim2real.arm-residual-environment/1",
                    config=self.config,
                    replay_sha256=digest(self.directory / "result.json"),
                    policy_steps=len(self.rewards),
                    rewards=self.rewards,
                    episode_return=float(sum(self.rewards)),
                    success=self.report["success"],
                    terminated=True,
                    truncated=False,
                    seed=self.seed,
                    deterministic_reset=True,
                ),
            )
        return self.packet["observation"].copy(), reward, terminated, truncated, info

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
            if not self.finished:
                trace_sha = (
                    self.bridge.save_trace(self.directory)
                    if self.bridge.packets
                    else None
                )
                write_json(
                    self.directory / "interrupted.json",
                    dict(
                        reason="caller_closed",
                        success=False,
                        training_eligible=False,
                        policy_trace_sha256=trace_sha,
                    ),
                )
                self.finished = True
