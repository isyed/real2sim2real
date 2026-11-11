import numpy as np
import pytest
from real2sim2real.controller_actor_training.finger_contact_trim import (
    FingerContactTrim,
    bind,
)


def settings():
    return dict(
        joint_directions={"pinky_PIP": 1, "pinky_DIP": -1},
        contact_group="pinky",
        target_force_n=1,
        minimum_anchor_force_n=0.1,
        gain_rad_per_n_s=0.02,
        max_speed_rad_s=0.01,
        max_offset_rad=0.08,
        maximum_reference_error_m=0.02,
    )


def controller():
    return FingerContactTrim(
        settings(),
        ["pinky_PIP", "pinky_DIP"],
        {"pinky_PIP": [-1, 1], "pinky_DIP": [-1, 1]},
    )


def update(c, **extra):
    args = dict(
        force_n=0, anchor_force_n=1, reference_error_m=0.005, active=True, dt=0.01
    )
    args.update(extra)
    return c.target([0, 0], **args)


def test_rate_travel_and_contact_recovery_freeze():
    c = controller()
    last = np.zeros(2)
    for _ in range(1000):
        q = update(c)
        assert np.linalg.norm(q - last) <= 0.00010000001
        assert np.linalg.norm(q) <= 0.08000000001
        last = q.copy()
    assert q[0] > 0 and q[1] < 0
    for args in [
        dict(force_n=2),
        dict(anchor_force_n=0),
        dict(reference_error_m=0.03),
        dict(active=False),
    ]:
        np.testing.assert_array_equal(update(c, **args), last)


def test_invalid_joint_state_and_ownership():
    with pytest.raises(ValueError, match="owned"):
        FingerContactTrim(settings(), ["arm"], {"arm": [-1, 1]})
    c = controller()
    with pytest.raises(ValueError):
        update(c, force_n=np.nan)
    with pytest.raises(ValueError, match="already satisfy"):
        c.target(
            [2, 0],
            force_n=0,
            anchor_force_n=1,
            reference_error_m=0,
            active=True,
            dt=0.01,
        )


def test_bind_excludes_other_hand_arm_and_object():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
    <body name="arm"><joint name="arm_joint" range="-1 1"/><geom size=".1"/>
    <body name="hand"><body><joint name="pinky_PIP" range="-1 1"/><geom size=".01"/>
    <body><joint name="pinky_DIP" range="-1 1"/><geom size=".01"/></body></body></body></body>
    <body name="object" pos="0 0 1"><freejoint/><geom size=".02"/></body>
    </worldbody><actuator><position joint="arm_joint" kp="10"/><position joint="pinky_PIP" kp="2"/><position joint="pinky_DIP" kp="2"/></actuator></mujoco>""")
    c, ids = bind(model, settings(), [0, 1, 2], model.body("hand").id)
    np.testing.assert_array_equal(ids, [1, 2])
    data = mj.MjData(model)
    before = data.qpos.copy()
    data.ctrl[ids] = update(c)
    np.testing.assert_array_equal(data.qpos, before)
    assert data.ctrl[0] == 0


def test_interpolated_command_at_stop_tolerates_only_roundoff():
    c = controller()
    q = c.target(
        [1 + np.finfo(float).eps, 0],
        force_n=2,
        anchor_force_n=1,
        reference_error_m=0,
        active=True,
        dt=0.01,
    )
    assert q[0] == 1
    with pytest.raises(ValueError, match="already satisfy"):
        c.target(
            [1.00001, 0],
            force_n=2,
            anchor_force_n=1,
            reference_error_m=0,
            active=True,
            dt=0.01,
        )
