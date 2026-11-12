"""Analytic loads check units, cone bounds, moment arms and motor limits."""

import numpy as np
import pytest
from real2sim2real.controller_actor_training.grasp_wrench_audit import (
    reference_kinematics,
)
from real2sim2real.controller_actor_training.wrench_feasibility import (
    build,
    contact_basis,
    solve,
)
from scipy.spatial.transform import Rotation


def pinch(dim=3, torque_limit=0.5):
    contacts = []
    for i, sign in enumerate((1, -1)):
        frame = np.diag([-sign, 1, 1])
        torque = np.zeros((2, dim))
        torque[i, 0] = 0.05
        contacts.append(
            {
                "wrench": contact_basis(frame, [sign * 0.05, 0, 0], [0, 0, 0], dim),
                "torque": torque,
                "friction": np.array([0.8, 0.8, 0.005, 0.001, 0.001])[: dim - 1],
                "group": str(i),
            }
        )

    def problem(cone):
        return build(
            contacts,
            np.zeros(2),
            np.full(2, -torque_limit),
            np.full(2, torque_limit),
            {"0": 10, "1": 10},
            cone,
        )

    return problem


def test_opposing_pinch_supports_known_weight_and_capacity_bounds():
    problem = pinch()
    demand = [0, 0, 5, 0, 0, 0]
    for cone in ("inner", "outer"):
        result = solve(problem(cone), demand)
        assert result["verified"]
        assert result["total_normal_force_n"] == pytest.approx(6.25)
        capacity = solve(problem(cone), demand, capacity=True)
        assert capacity["capacity_factor"] == pytest.approx(3.2)


def test_torque_limits_can_make_frictionally_possible_grasp_impossible():
    problem = pinch(torque_limit=0.05)
    demand = [0, 0, 5, 0, 0, 0]
    assert solve(problem("outer"), demand)["status"] == 2
    assert solve(problem("outer"), demand, capacity=True)[
        "capacity_factor"
    ] == pytest.approx(0.32)


def test_torsional_friction_is_needed_for_moment_about_pinch_axis():
    demand = [0, 0, 0, 0.04, 0, 0]
    assert solve(pinch(3)("outer"), demand)["status"] == 2
    result = solve(pinch(4)("inner"), demand)
    assert result["verified"] and result["exact_cone_satisfied"]
    assert result["total_normal_force_n"] == pytest.approx(8)


def test_cone_inner_and_outer_bracket_oblique_load_capacity():
    demand = [0, 1.1, 5, 0, 0, 0]
    problem = pinch()
    inner = solve(problem("inner"), demand, capacity=True)
    outer = solve(problem("outer"), demand, capacity=True)
    exact = 16 / np.hypot(1.1, 5)
    assert inner["capacity_factor"] <= exact + 1e-8
    assert outer["capacity_factor"] >= exact - 1e-8
    assert inner["exact_cone_satisfied"]


def test_contact_moments_use_com_and_are_translation_invariant():
    B = contact_basis(np.eye(3), [1, 2, 3], [0.9, 2, 3], 6)
    translated = contact_basis(np.eye(3), [8, 0, 6], [7.9, 0, 6], 6)
    np.testing.assert_allclose(B, translated, atol=1e-14)
    np.testing.assert_allclose(B @ [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0.1])
    np.testing.assert_allclose(B @ [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0])


def test_reference_derivatives_recover_known_acceleration_and_world_rotation():
    times = np.linspace(0, 2, 41)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, 0, 3] = 0.5 * 0.3 * times**2
    poses[:, :3, :3] = Rotation.from_rotvec(
        np.c_[times * 0, times * 0, 0.5 * 0.2 * times**2]
    ).as_matrix()
    result = reference_kinematics(times, poses, 1, np.zeros(3), 7)
    np.testing.assert_allclose(
        result["acceleration_world_m_s2"], [0.3, 0, 0], atol=1e-10
    )
    np.testing.assert_allclose(result["omega_world_rad_s"], [0, 0, 0.2], atol=1e-8)
    np.testing.assert_allclose(result["alpha_world_rad_s2"], [0, 0, 0.2], atol=1e-7)


def test_invalid_torque_envelope_is_rejected():
    with pytest.raises(ValueError, match="torque envelope"):
        build([], [0], [1], [-1], {}, "inner")


def test_maximum_load_above_one_does_not_imply_nominal_load_is_feasible():
    # A lower actuator torque bound requires at least 2 N at this contact.
    contact = {
        "wrench": np.array([[0], [0], [1], [0], [0], [0]]),
        "torque": np.array([[0.1]]),
        "friction": np.array([]),
        "group": "pad",
    }
    problem = build([contact], [0], [0.2], [1], {"pad": 10}, "inner")
    assert solve(problem, [0, 0, 1, 0, 0, 0])["status"] == 2
    assert solve(problem, [0, 0, 1, 0, 0, 0], capacity=True)[
        "capacity_factor"
    ] == pytest.approx(10)


def test_saturated_servo_range_is_clipped_instead_of_wrongly_intersected():
    from types import SimpleNamespace

    from real2sim2real.controller_actor_training.grasp_wrench_audit import torque_bounds

    m = SimpleNamespace(
        jnt_qposadr=np.array([0]),
        actuator_gainprm=np.array([[1.0]]),
        actuator_biasprm=np.array([[0.0, 0.0, 0.0]]),
        actuator_forcerange=np.array([[-2.0, 2.0]]),
        jnt_actfrclimited=np.array([False]),
        actuator_ctrllimited=np.array([True]),
        actuator_ctrlrange=np.array([[3.0, 4.0]]),
    )
    d = SimpleNamespace(
        qpos=np.array([0.0]), qvel=np.array([0.0]), ctrl=np.array([3.5])
    )
    layout = {
        "ids": np.array([0]),
        "joints": np.array([0]),
        "dofs": np.array([0]),
        "members": {},
    }
    lo, hi, *_ = torque_bounds(m, d, layout, {}, "motor", 0.2)
    assert lo[0] == 2 and hi[0] == 2
