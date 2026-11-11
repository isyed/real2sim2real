import json
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.residual_pressure import (
    CHANNELS,
    PressureResidual,
    pressure_goal,
    validate_action,
    validate_offsets,
)
from real2sim2real.controller_actor_training.arm_residual_env import (
    ArmResidualEnv,
    PressureBridge,
    frozen_source,
)
from real2sim2real.controller_actor_training.arm_residual_observation import (
    transition_reward,
)


def settings():
    return dict(
        policy_period_s=0.02, max_offset_n=1.0, max_rate_n_s=5.0, active_phases=["hold"]
    )


def test_pressure_residual_is_zero_preserving_bounded_and_causal():
    control = PressureResidual(settings())
    assert control.advance(0.001, "hold") is None
    assert pressure_goal(0.05, 0.0) == 0.05  # No new clamp on the original zero path.
    action = np.array([1.0, -1.0, 0.5, -0.5])
    control.accept(action)
    action[:] = 0  # Caller cannot mutate the pending action.
    assert control.advance(0.001, "approach") is None
    for _ in range(400):
        previous = control.offset.copy()
        control.advance(0.001, "hold")
        assert np.max(abs(control.offset - previous)) <= 0.005 + 1e-12
        assert np.max(abs(control.offset)) <= 1
    np.testing.assert_allclose(control.offset, [1, -1, 0.5, -0.5])
    assert pressure_goal(9.8, 1.0) == 10
    assert pressure_goal(0.2, -1.0) == 0.1
    for _ in range(400):
        control.advance(0.001, "approach")
    assert control.advance(0.001, "approach") is None


@pytest.mark.parametrize(
    "action", [[1, 2, 3, 4], [0, 0], [0, np.nan, 0, 0], [0, np.inf, 0, 0]]
)
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        validate_action(action)


def test_unknown_settings_offsets_and_unsafe_budgets_rejected():
    for changes in [
        dict(max_offset_n=3),
        dict(max_rate_n_s=11),
        dict(policy_period_s=0.001),
        dict(active_phases=["unknown"]),
        dict(foo=1),
    ]:
        with pytest.raises(ValueError):
            PressureResidual(dict(settings(), **changes))
    with pytest.raises(ValueError):
        validate_offsets({"unknown": 0})
    with pytest.raises(ValueError):
        validate_offsets(dict.fromkeys(CHANNELS, np.nan))
    with pytest.raises(ValueError):
        pressure_goal(10, 3)


def test_learning_reward_preserves_time_scale_and_terminal_failures():
    costs = dict(position=1, rotation=2, grasp=3, contact=1, excess_force=0, residual=0)
    previous = dict(time_s=0)
    current = dict(time_s=0.02, costs=costs)
    reward = transition_reward(previous, current, horizon_s=1)
    half = transition_reward(previous, dict(current, time_s=0.01), horizon_s=1)
    assert reward == 2 * half
    assert (
        transition_reward(previous, current, horizon_s=1, success=False) == reward - 10
    )
    assert (
        transition_reward(previous, current, horizon_s=1, success=True) == reward + 10
    )
    assert (
        transition_reward(previous, current, horizon_s=1, failure=True) == reward - 110
    )
    with pytest.raises(ValueError):
        transition_reward(current, current, horizon_s=1)


def frozen_fixture(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "scene.xml").write_text("<mujoco/>")
    np.savez(
        baseline / "states.npz", qpos=[[0]], ctrl=[[0]], qvel=[[0]], timestamps_s=[0]
    )
    source = dict(
        schema="real2sim2real.arm-replay-pilot/1",
        config=dict(output=str(baseline)),
        finger_control_mode="live_contact_force",
        physics_run=True,
        fixed_controller_clock=True,
        rejected_for=["hold_slip"],
        success=False,
        states_sha256=digest(baseline / "states.npz"),
        scene_sha256=digest(baseline / "scene.xml"),
    )
    path = baseline / "result.json"
    path.write_text(json.dumps(source))
    config = dict(
        baseline_result=str(path),
        baseline_sha256=digest(path),
        output=str(tmp_path / "env"),
        residual=settings(),
        deadline_s=60,
    )
    return config, source


def test_pinned_failed_baseline_allowed_but_mutations_and_stacking_reject(tmp_path):
    config, source = frozen_fixture(tmp_path)
    assert frozen_source(config)["success"] is False
    for change in [
        dict(residual_adapter={"enabled": True}),
        dict(rejected_for=["incomplete_pilot"]),
        dict(fixed_controller_clock=False),
        dict(continuous_protocol={"schema": "new_phase_semantics"}),
    ]:
        path = tmp_path / "baseline/result.json"
        path.write_text(json.dumps(dict(source, **change)))
        modified = dict(config, baseline_sha256=digest(path))
        with pytest.raises(ValueError):
            frozen_source(modified)
    with pytest.raises(ValueError, match="changed"):
        frozen_source(config)


def test_reset_step_terminal_and_invalid_actions_without_advancing(
    tmp_path, monkeypatch
):
    from real2sim2real.controller_actor_training import arm_replay_pilot

    config, _ = frozen_fixture(tmp_path)
    received = []

    def rollout(config, interaction):
        output = __import__("pathlib").Path(config["output"])
        output.mkdir()
        interaction.observer = SimpleNamespace(schema={"state": [0, 2]})
        interaction.horizon_s = 0.04
        costs = dict.fromkeys(
            ("position", "rotation", "grasp", "contact", "excess_force", "residual"),
            0.0,
        )
        for i in range(2):
            packet = dict(
                observation=np.array([float(i), 0]),
                phase="hold",
                time_s=i * 0.02,
                costs=costs,
            )
            received.append((yield packet))
        interaction.packets = [dict(packet, time_s=0.04)]
        report = dict(success=False, rejected_for=["hold_slip"])
        (output / "result.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(arm_replay_pilot, "rollout", rollout)
    env = ArmResidualEnv(config)
    obs, info = env.reset(seed=5)
    assert env.observation_dim == 2 and info["privileged_observations"]
    obs[:] = 999
    with pytest.raises(ValueError):
        env.step([np.nan] * 4)
    assert not received
    obs, _, terminal, truncated, _ = env.step(np.zeros(4))
    assert obs[0] == 1 and not terminal and not truncated
    _, reward, terminal, truncated, info = env.step(np.zeros(4))
    assert terminal and not truncated and not info["success"] and reward == -10
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))
    obs, _ = env.reset()
    assert obs[0] == 0
    env.close()
    assert (
        json.loads((env.directory / "interrupted.json").read_text())["success"] is False
    )


def test_policy_period_must_match_controller_ticks():
    bridge = PressureBridge(dict(settings(), policy_period_s=0.0205), 60)
    with pytest.raises(ValueError, match="integer"):
        bridge.bind(None, None, None, None, None, None, 0.001)


def test_hold_anchor_is_fixed_and_uses_the_evaluator_entry_sample():
    from real2sim2real.controller_actor_training.arm_residual_observation import (
        ArmResidualObservation,
    )

    observer = ArmResidualObservation.__new__(ArmResidualObservation)
    observer.anchors = None
    sample = dict(
        phase="lift",
        object_position_m=[0.2, 0.3, 0.4],
        wrist_position_m=[0.1, 0, 0],
        support_wrist_position_m=[0, 0.1, 0],
        wrist_rotation=np.eye(3),
        support_wrist_rotation=np.eye(3),
    )
    observer.record(sample)
    assert observer.anchors is None
    sample["phase"] = "hold"
    observer.record(sample)
    original = np.array(observer.anchors).copy()
    sample["object_position_m"] = [99, 99, 99]
    observer.record(sample)
    np.testing.assert_array_equal(observer.anchors, original)


def test_timeout_is_truncated_and_cannot_be_scored_as_success(tmp_path, monkeypatch):
    from real2sim2real.controller_actor_training import arm_replay_pilot

    config, _ = frozen_fixture(tmp_path)

    def rollout(config, interaction):
        from pathlib import Path

        Path(config["output"]).mkdir()
        interaction.observer = SimpleNamespace(schema={})
        yield dict(observation=np.zeros(2), phase="hold", time_s=0.0, costs={})
        raise AssertionError("Expired episode must not advance")

    monkeypatch.setattr(arm_replay_pilot, "rollout", rollout)
    env = ArmResidualEnv(config)
    env.reset()
    env.bridge.deadline = -1
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert not terminated and truncated and not info["training_eligible"]
    assert not (env.directory / "result.json").exists()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))
    env.close()
