from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.payload_pressure import PayloadPressure


def settings():
    return dict(
        reference_weight_n=4.905,
        weight_filter_s=0.15,
        share_filter_s=0.15,
        minimum_weight_ratio=0.5,
        maximum_weight_ratio=1.25,
        target_secondary_fraction=0.5,
        share_gain=0.5,
        maximum_share_correction=0.15,
        maximum_scale_rate_s_inv=1,
        maximum_acceleration_m_s2=3,
    )


def test_load_estimate_compensates_acceleration_and_requires_isolated_support():
    c = PayloadPressure(settings())
    for _ in range(1500):
        c.update([2.5, 2.5], 0, 9.81, 0.002, active=True, isolated=False)
    assert c.weight_n == 4.905 and c.valid_updates == 0
    # 0.375 kg accelerated upward at 2 m/s²; Fz is not just weight.
    force = 0.375 * (9.81 + 2)
    for _ in range(1500):
        c.update([force / 2, force / 2], 2, 9.81, 0.002, active=True, isolated=True)
    assert c.weight_n == pytest.approx(0.375 * 9.81, abs=1e-7)
    np.testing.assert_allclose(c.scales, [0.75, 0.75], atol=1e-7)
    assert c.target("primary", 8) == pytest.approx(6, abs=1e-6)


def test_share_correction_has_opposite_sign_and_preserves_pair_pressure_budget():
    c = PayloadPressure(settings())
    for _ in range(2000):
        c.update([3.5, 1.405], 0, 9.81, 0.002, active=True, isolated=True)
    assert c.scales[0] < 1 < c.scales[1]
    assert c.scales.sum() == pytest.approx(2)
    assert abs(c.scales[1] - 1) <= 0.15


def test_rate_caps_force_caps_and_invalid_observation_freeze():
    c = PayloadPressure(settings())
    c.update([4, 4], 0, 9.81, 0.001, active=True, isolated=True)
    assert abs(c.scales - 1).max() <= 0.001 + 1e-12
    before = c.scales.copy()
    c.update([4, 4], 10, 9.81, 0.01, active=True, isolated=True)
    np.testing.assert_array_equal(c.scales, before)
    for _ in range(3000):
        c.update([1, 8], 0, 9.81, 0.002, active=True, isolated=True)
    assert c.target("primary", 10) <= 10 and c.target("secondary", 10) <= 10
    assert c.saturated_targets > 0
    bad = deepcopy(settings())
    bad["share_gain"] = float("nan")
    with pytest.raises(ValueError):
        PayloadPressure(bad)
    with pytest.raises(ValueError):
        c.update([float("nan"), 0], 0, 9.81, 0.002, active=True, isolated=True)


def test_anatomical_floor_can_preserve_opposing_pressure_while_releasing_thumb():
    s = settings()
    s["minimum_group_ratio"] = dict(thumb=0.5, index=1, middle=1, ring=1, pinky=1)
    c = PayloadPressure(s)
    for _ in range(2000):
        c.update([1.839375, 1.839375], 0, 9.81, 0.002, active=True, isolated=True)
    assert c.target("primary", 8, group="thumb") == pytest.approx(6, abs=1e-5)
    assert c.target("primary", 4, group="index") == 4
    with pytest.raises(ValueError, match="anatomical group"):
        c.target("primary", 4)
