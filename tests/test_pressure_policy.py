from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.pressure_policy import PressurePolicy
from real2sim2real.controller_actor_training.arm_residual_trial import matched_protocol


def test_constant_policy_copies_requests_and_does_not_need_future_state():
    settings = dict(kind="constant", action=[0.5, -0.5, 0, 0])
    policy = PressurePolicy(settings)
    settings["action"][0] = 1
    action = policy(None, None)
    np.testing.assert_array_equal(action, [0.5, -0.5, 0, 0])
    action[:] = 0
    assert policy(None, None)[0] == 0.5


def test_drift_feedback_requires_fixed_anchor_and_uses_current_primary_drift():
    policy = PressurePolicy(
        dict(kind="drift_ramp", action=[1, 0.5, 0, 0], start_m=0.004, full_m=0.008)
    )
    info = dict(
        observation_schema=dict(
            grasp_anchor_valid=[0, 1], grasp_drift_in_wrists_m=[1, 7]
        )
    )
    obs = np.array([0, 0.006, 0, 0, 0.009, 0, 0])
    np.testing.assert_array_equal(policy(obs, info), np.zeros(4))
    obs[0] = 1
    np.testing.assert_allclose(policy(obs, info), [0.5, 0.25, 0, 0])
    obs[1] = 0.02
    np.testing.assert_array_equal(policy(obs, info), [1, 0.5, 0, 0])
    obs[1] = 0.001
    np.testing.assert_array_equal(policy(obs, info), np.zeros(4))


@pytest.mark.parametrize(
    "settings",
    [
        dict(kind="constant", action=[2, 0, 0, 0]),
        dict(kind="constant", action=[0, 0, 0, 0], future=True),
        dict(kind="drift_ramp", action=[0] * 4, start_m=0.008, full_m=0.004),
        dict(kind="drift_ramp", action=[0] * 4, start_m=0.004, full_m=float("nan")),
        dict(kind="unknown", action=[0] * 4),
    ],
)
def test_policy_rejects_unsafe_or_undeclared_settings(settings):
    with pytest.raises(ValueError):
        PressurePolicy(settings)


def test_matching_checks_reject_gate_changes_clock_changes_and_truncation():
    baseline = dict(
        config=dict(output="a", limits=dict(slip=0.01)),
        scene_sha256="x",
        controller_period_s=0.001,
        physics_timestep_s=0.000125,
        records=[dict(time_s=1, phase="hold")],
        controller_updates=10,
        rejected_for=["hold_slip"],
    )
    candidate = deepcopy(baseline)
    candidate["config"]["output"] = "b"
    assert all(matched_protocol(baseline, candidate).values())
    candidate["config"]["limits"]["slip"] = 0.02
    assert not matched_protocol(baseline, candidate)["frozen_config_equal"]
    candidate = deepcopy(baseline)
    candidate["controller_period_s"] = 0.002
    assert not matched_protocol(baseline, candidate)["controller_clock_equal"]
    candidate = deepcopy(baseline)
    candidate["records"] = []
    assert not matched_protocol(baseline, candidate)["complete_timeline"]
