import numpy as np
import pytest
from real2sim2real.controller_actor_training.table_load_transfer import (
    TableLoadTransfer,
)


def settings(**kwargs):
    return dict(
        gain_m_per_n_s=0.001,
        max_offset_m=0.006,
        max_speed_m_s=0.012,
        confirmation_s=0.04,
        support_fraction=0.5,
        deadband_fraction=0.1,
        **kwargs,
    )


def make(s=None):
    return TableLoadTransfer(
        s or settings(), {"capture_duration_s": 4.0, "finger_release": {"start_s": 5.0}}
    )


def test_only_confirmed_table_support_in_return_can_add_common_upward_motion():
    c = make()
    for phase, contacts in [("hold", 4), ("return", 0)]:
        for i in range(100):
            assert (
                c.advance(
                    2 + i * 0.001, 0.001, 20, 5, table_contacts=contacts, phase=phase
                )
                == 0
            )
    for i in range(39):
        assert (
            c.advance(3 + i * 0.001, 0.001, 20, 5, table_contacts=4, phase="return")
            == 0
        )
    previous = 0
    for i in range(39, 600):
        c.advance(3 + i * 0.001, 0.001, 20, 5, table_contacts=4, phase="return")
        offsets = c.compose(np.zeros((2, 6)), 0.006)
        assert 0 <= offsets[0, 2] - previous <= 0.000012 + 1e-12
        assert offsets[0, 2] == offsets[1, 2] <= 0.006
        assert not offsets[:, [0, 1, 3, 4, 5]].any()
        previous = offsets[0, 2]
    assert c.first_active_s == pytest.approx(3.039)


def test_projection_shares_original_budget_preserves_lateral_rotation_and_antiwindup():
    c = make()
    c.offset = 0.006
    original = np.zeros((2, 6))
    original[0] = [0.003, 0, 0.004, 0.02, 0.01, 0]
    before = original.copy()
    result = c.compose(original, 0.006)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(
        result[:, [0, 1, 3, 4, 5]], original[:, [0, 1, 3, 4, 5]]
    )
    assert max(np.linalg.norm(result[:, :3], axis=1)) <= 0.006 + 1e-12
    assert result[0, 2] - original[0, 2] == pytest.approx(result[1, 2])
    assert c.offset == pytest.approx(np.sqrt(0.006**2 - 0.003**2) - 0.004)
    assert c.budget_limited_steps == 1


def test_support_loss_recovers_and_release_returns_exactly_to_original_without_opening():
    c = make()
    c.offset = 0.005
    c.advance(3, 0.01, 0, 5, table_contacts=0, phase="return")
    assert c.offset == pytest.approx(0.00488)
    initial = c.offset
    assert c.advance(4, 0.01, 0, 5, table_contacts=0, phase="settle") == initial
    assert c.advance(
        4.5, 0.01, 0, 5, table_contacts=0, phase="settle"
    ) == pytest.approx(initial * 0.5)
    assert c.advance(5, 0.01, 0, 5, table_contacts=0, phase="release") == 0
    original = np.zeros((2, 6))
    np.testing.assert_array_equal(c.compose(original, 0.006), original)


def test_zero_gain_never_changes_offsets_and_unbounded_settings_fail():
    s = settings()
    s["gain_m_per_n_s"] = 0
    c = make(s)
    for i in range(100):
        assert (
            c.advance(3 + i * 0.001, 0.001, 20, 5, table_contacts=4, phase="return")
            == 0
        )
    for key, value in [
        ("gain_m_per_n_s", 0.1),
        ("max_offset_m", 0.007),
        ("confirmation_s", 0),
    ]:
        s = settings()
        s[key] = value
        with pytest.raises(ValueError, match="bounded"):
            make(s)


def test_arm_command_capture_uses_scratch_and_distinguishes_servo_feedforward():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.arm_command_observation import capture

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="arm"><joint name="q" type="hinge" axis="0 0 1"/><geom type="sphere" size=".1"/><body name="wrist" pos="1 0 0"/></body></worldbody><actuator><position joint="q" kp="10"/></actuator></mujoco>'
    )
    d, scratch = mujoco.MjData(m), mujoco.MjData(m)
    d.qpos[:] = 0.3
    d.qvel[:] = 0.4
    d.ctrl[:] = 0.7
    mujoco.mj_forward(m, d)
    before = [getattr(d, k).copy() for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
    command = {
        "command_time_s": 0.0,
        "planned_q_rad": np.array([0.1]),
        "kinematic_q_rad": np.array([0.2]),
        "servo_ctrl_rad": np.array([0.7]),
    }
    result = capture(
        m, d, scratch, {"right": m.body("wrist").id}, np.array([0]), command
    )
    for k, old in zip(("qpos", "qvel", "ctrl", "qacc_warmstart"), before):
        np.testing.assert_array_equal(getattr(d, k), old)
    for label, angle in [("planned", 0.1), ("commanded", 0.2), ("actual", 0.3)]:
        np.testing.assert_allclose(
            np.array(result["wrists"]["right"][label])[:3, 3],
            [np.cos(angle), np.sin(angle), 0],
        )
    assert result["servo_ctrl_rad"] != result["kinematic_q_rad"]


def test_table_observation_excludes_downward_pressure_from_other_geometry():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.support_control import (
        contact_wrench_on_body,
    )
    from real2sim2real.controller_actor_training.table_load_transfer import (
        observe_table_force,
    )

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/><geom name="hand" type="sphere" pos="0 0 .205" size=".03"/><body name="object" pos="0 0 .09"><freejoint/><geom name="object_geom" type="box" size=".1 .1 .1" mass=".5"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    obj, table, geom = m.body("object").id, m.geom("table").id, m.geom("object_geom").id
    force, count = observe_table_force(m, d, {geom}, table, obj)
    total = sum(
        contact_wrench_on_body(m, d, cid, obj)[0][2]
        for cid, c in enumerate(d.contact)
        if geom in (c.geom1, c.geom2)
    )
    assert force > 0 and count > 0
    assert total < force
