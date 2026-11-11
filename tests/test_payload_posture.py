from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.payload_posture import (
    BoundedPostureFraction,
)


def settings():
    return dict(
        target_result="admitted.json",
        target_sha256="pinned",
        activation_weight_ratio=0.95,
        full_weight_ratio=0.8,
        maximum_joint_change_rad=0.35,
        maximum_joint_speed_rad_s=0.2,
        maximum_fraction_speed_s_inv=1.0,
    )


def test_nominal_heavy_deadband_and_light_transition_speed():
    p = BoundedPostureFraction(settings(), 0.25)
    for ratio in [1, 1.25, 0.96]:
        assert p.update(ratio, 0.01, active=True) == 0
    previous = p.fraction
    for _ in range(200):
        f = p.update(0.75, 0.01, active=True)
        assert 0 <= f <= 1 and abs(f - previous) * 0.25 <= 0.2 * 0.01 + 1e-12
        previous = f
    assert p.fraction == 1
    p.update(1, 0.01, active=False)
    assert p.fraction == 1
    for _ in range(200):
        p.update(1, 0.01, active=True)
    assert p.fraction == 0


def test_intermediate_target_is_continuous_and_invalid_input_rejected():
    p = BoundedPostureFraction(settings(), 0.2)
    for _ in range(200):
        p.update(0.875, 0.01, active=True)
    assert p.fraction == pytest.approx(0.5)
    bad = deepcopy(settings())
    bad["full_weight_ratio"] = 1
    with pytest.raises(ValueError):
        BoundedPostureFraction(bad, 0.2)
    with pytest.raises(ValueError):
        BoundedPostureFraction(settings(), 0.4)
    with pytest.raises(ValueError):
        p.update(float("nan"), 0.01, active=True)


def test_apply_owns_fingers_and_resets_patch_when_returning_to_baseline():
    from real2sim2real.controller_actor_training.payload_posture import PayloadPosture

    c = PayloadPosture.__new__(PayloadPosture)
    c.policy = BoundedPostureFraction(settings(), 0.2)
    c.owned = np.array([1, 3])
    c.delta = np.array([0.2, -0.1])
    c.limits = np.array([[-1, 1], [-1, 1]])
    c.grid = np.linspace(0, 1, 33)
    c.points = {7: np.c_[c.grid, c.grid * 0, c.grid * 0]}
    c.clipped_steps = 0
    pads = [dict(gid=7, point_body=np.zeros(3))]
    nominal = np.arange(6, dtype=float) * 0.01
    original = nominal.copy()
    for _ in range(200):
        result = c.apply(nominal, pads, 0.75, 0.01, active=True)
    np.testing.assert_array_equal(nominal, original)
    np.testing.assert_array_equal(result[[0, 2, 4, 5]], original[[0, 2, 4, 5]])
    np.testing.assert_allclose(result[c.owned], nominal[c.owned] + c.delta)
    for _ in range(200):
        result = c.apply(nominal, pads, 1.0, 0.01, active=True)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(pads[0]["point_body"], [0, 0, 0])


@pytest.mark.parametrize("failed", ["physical", "incomplete", "nonfinite"])
def test_endpoint_must_be_a_complete_admitted_trial(tmp_path, failed):
    import json
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.payload_posture import PayloadPosture

    directory = tmp_path / "trial"
    directory.mkdir()
    path = directory / "result.json"
    target = dict(
        success=True,
        candidate=dict(accepted=True),
        simulation_final_time_s=10.0,
        simulation_requested_duration_s=10.0,
    )
    if failed == "physical":
        target["success"] = False
    if failed == "incomplete":
        target["simulation_final_time_s"] = 2.0
    if failed == "nonfinite":
        target["simulation_final_time_s"] = float("inf")
    path.write_text(json.dumps(target))
    (tmp_path / "protocol.json").write_text("{}")
    s = settings()
    s.update(target_result=str(path), target_sha256=digest(path))
    with pytest.raises(ValueError, match="admitted nominal posture"):
        PayloadPosture(
            None, None, s, {}, [], scene_sha256=None, allowed_penetration_m=0.003
        )
