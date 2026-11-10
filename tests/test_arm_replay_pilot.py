import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_replay_pilot import (
    motor_command,
    subtree_geoms,
    bias_compensated_target,
)


def test_bias_compensation_is_inside_torque_budget():
    torque, clipped = motor_command(
        [1, -1], [0, 0], [2, -2], [100, -100], 10, 1, [5, 6]
    )
    np.testing.assert_allclose(torque, [5, -6])
    assert clipped.all()
    with pytest.raises(ValueError):
        motor_command([np.nan], [0], [0], [0], 10, 1, [5])
    with pytest.raises(ValueError):
        motor_command([0], [0], [0], [0], -1, 1, [5])


def test_hand_contacts_exclude_its_parent_arm_and_free_object():
    from types import SimpleNamespace

    model = SimpleNamespace(
        ngeom=4, geom_bodyid=[1, 2, 3, 4], body_parentid=[0, 0, 1, 2, 0]
    )
    assert subtree_geoms(model, 2) == {1, 2}
    assert subtree_geoms(model, 4) == {3}


def test_implicit_pd_bias_and_velocity_feedforward_remain_force_limited():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string("""<mujoco><option integrator="implicitfast"/>
    <worldbody><body><joint name="arm"/><geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="0 0 2"><freejoint name="object"/><geom size=".03"/></body></worldbody>
    <actuator><position joint="arm" kp="600" kv="45" forcerange="-2 2"/></actuator></mujoco>""")
    d = mujoco.MjData(m)
    target = bias_compensated_target(
        np.array([1.0]),
        np.array([10.0]),
        600.0,
        target_velocity=np.array([2.0]),
        kd=45.0,
    )
    np.testing.assert_allclose(target, [1 + 100 / 600])
    d.ctrl[:] = target
    before = d.qpos[m.joint("object").qposadr[0] + 2]
    for _ in range(50):
        mujoco.mj_step(m, d)
        assert abs(d.actuator_force[0]) <= 2.000001
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
    assert d.qpos[m.joint("object").qposadr[0] + 2] < before - 0.01
