import numpy as np
import pytest

from real2sim2real.controller_actor_training.descent_pad_control import (
    DescentPads,
    LABELS,
)


def settings():
    return dict(
        normal_labels=list(LABELS),
        anchor_labels=[],
        anchor_settings=None,
        rebalance_n={
            "primary:thumb": -1,
            "primary:middle": 0.5,
            "primary:ring": 0.5,
            "secondary:thumb": -1,
            "secondary:index": 0.5,
            "secondary:pinky": 0.5,
        },
        share_gain_n=1.5,
        maximum_share_offset_n=0.5,
        maximum_offset_n=1,
        maximum_offset_rate_n_s=2,
        ramp_s=0.4,
    )


def make(s=None):
    return DescentPads(
        s or settings(),
        dict(acquisition_end_s=1, capture_duration_s=4, finger_release=dict(start_s=5)),
        2,
    )


def test_pair_pressure_sum_rate_bounds_and_signed_load_sharing():
    p = make()
    old = np.zeros(10)
    for _ in range(600):
        p.advance(2.5, 0.001, [4, 1], active=True, isolated=True)
        assert abs(p.offset.sum()) < 1e-12
        assert max(abs(p.offset - old)) <= 0.002 + 1e-12
        assert max(abs(p.offset)) <= 1 + 1e-12
        old = p.offset.copy()
    assert p.offset[5:].sum() > 0 and p.offset[:5].sum() < 0
    targets = [p.target(n, 8 if n.endswith(":thumb") else 4) for n in LABELS]
    assert sum(targets) == pytest.approx(48)
    p.advance(2.5, 0.001, [-2, 5], active=True, isolated=True)
    assert p.secondary_fraction == 1
    previous = p.offset.copy()
    p.advance(2.5, 0.001, [4, 1], active=True, isolated=False)
    assert np.linalg.norm(p.offset) < np.linalg.norm(previous)


def test_prefix_and_zero_policy_preserve_original_goals_without_new_clamps():
    p = make()
    p.advance(1.9, 0.001, [4, 1], active=True, isolated=True)
    assert not p.offset.any() and not p.observed_normal("primary:middle")
    assert p.target("primary:thumb", 0.05) == 0.05
    s = settings()
    s.update(normal_labels=[], rebalance_n={}, share_gain_n=0)
    p = make(s)
    for _ in range(10):
        p.advance(3, 0.001, [4, 1], active=True, isolated=True)
    assert not p.offset.any() and not p.observed_normal("primary:middle")
    assert p.target("primary:thumb", 0.05) == 0.05


def test_recovery_finishes_on_time_even_after_contact_loss():
    p = make()
    for _ in range(600):
        p.advance(3, 0.001, [4, 1], active=True, isolated=True)
    start = p.offset.copy()
    p.advance(4, 0.001, [0, 0], active=False, isolated=False)
    np.testing.assert_array_equal(p.offset, start)
    p.advance(4.5, 0.001, [0, 0], active=False, isolated=False)
    np.testing.assert_allclose(p.offset, start * 0.5)
    p.advance(5, 0.001, [0, 0], active=False, isolated=False)
    assert not p.offset.any() and p.amplitude == 0


def test_complete_stationary_hold_retains_the_descent_policy():
    p = make()
    p.protocol["release_disabled_for_stationary_diagnostic"] = True
    for _ in range(600):
        p.advance(3, 0.001, [4, 1], active=True, isolated=True)
    previous = p.offset.copy()
    p.advance(6, 0.001, [4, 1], active=True, isolated=True)
    np.testing.assert_allclose(p.offset, previous)
    assert p.amplitude == 1 and p.recovery is None


def test_material_anchor_latches_once_and_respects_contact_and_speed_guards():
    s = settings()
    s.update(
        anchor_labels=["primary:middle"],
        anchor_settings=dict(
            gain_s_inv=1.5,
            max_speed_m_s=0.002,
            minimum_force_n=0.5,
            maximum_error_m=0.01,
        ),
    )
    p = make(s)
    p.advance(2.5, 0.001, [2, 2], active=True, isolated=True)
    body, obj = np.eye(4), np.eye(4)
    obs = dict(point_world_m=np.zeros(3), force_n=1.0)
    normal = np.array([0.0, 0.0, 1.0])
    assert p.anchor_task("primary:index", body, obj, obs, normal) is None
    p.anchor_task("primary:middle", body, obj, obs, normal)
    original = p.anchors.anchors["primary:middle"]["object"].copy()
    obj[0, 3] = 0.001
    _, _, velocity = p.anchor_task("primary:middle", body, obj, obs, normal)
    assert velocity[0] > 0 and np.linalg.norm(velocity) <= 0.002
    p.anchor_task(
        "primary:middle", body, obj, dict(obs, point_world_m=np.ones(3) * 0.01), normal
    )
    np.testing.assert_array_equal(
        p.anchors.anchors["primary:middle"]["object"], original
    )
    p.isolated = False
    assert p.anchor_task("primary:middle", body, obj, obs, normal) is None


def test_invalid_budget_ownership_and_recovery_rate_reject():
    for change in [
        dict(rebalance_n={"primary:thumb": 1}),
        dict(normal_labels=["object:free"]),
        dict(maximum_offset_rate_n_s=0.1),
        dict(anchor_labels=["primary:thumb"]),
    ]:
        s = settings()
        s.update(change)
        with pytest.raises(ValueError):
            make(s)


def test_normal_only_seating_is_phase_scoped_and_has_exclusive_tangent_ownership():
    s = settings()
    s["zero_tangent_labels"] = ["primary:middle"]
    p = make(s)
    p.advance(1.9, 0.001, [2, 2], active=True, isolated=True)
    assert not p.zero_tangent("primary:middle")
    p.advance(2.5, 0.001, [2, 2], active=True, isolated=True)
    assert p.zero_tangent("primary:middle") and not p.zero_tangent("primary:thumb")
    p.isolated = False
    assert not p.zero_tangent("primary:middle")
    s["anchor_labels"] = ["primary:middle"]
    s["anchor_settings"] = dict(
        gain_s_inv=1.5, max_speed_m_s=0.002, minimum_force_n=0.5, maximum_error_m=0.01
    )
    with pytest.raises(ValueError, match="exclusive"):
        make(s)


def test_normal_pressing_task_can_suppress_uncommanded_tangent_motion():
    from real2sim2real.controller_actor_training.contact_force_control import (
        normal_force_step,
    )

    s = dict(
        gain_m_per_n_s=0.001,
        max_normal_speed_m_s=0.002,
        damping_m2=1e-8,
        max_joint_speed_rad_s=0.2,
        max_joint_offset_rad=0.2,
    )
    q = np.zeros(2)
    normal = np.array([1.0, 1.0])
    limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    legacy, _ = normal_force_step(q, q, normal, 0, 4, 0.01, limits, s)
    jac = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    seated, _ = normal_force_step(
        q,
        q,
        normal,
        0,
        4,
        0.01,
        limits,
        s,
        tangent_jacobian=jac,
        tangent_velocity=np.zeros(3),
    )
    assert abs(seated[0]) < abs(legacy[0]) * 0.01
    assert normal @ seated / 0.01 == pytest.approx(-0.002, abs=1e-8)
