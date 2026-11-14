import numpy as np
import pytest


def test_endpoint_reachability_does_not_hide_obstacle_or_mutate_live_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.arm_pose_screen import screen

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".05" pos="0 0 3"/>
      <body name="upper"><joint name="a" axis="0 0 1" range="-170 170"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".02"/>
      <body name="forearm" pos="1 0 0"><joint name="b" axis="0 0 1" range="-170 170"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".02"/>
      <body name="hand" pos="1 0 0"/></body></body>
      <body pos="0 0 2"><freejoint/><geom size=".03"/></body>
      </worldbody></mujoco>""")
    data = mj.MjData(model)
    data.qpos[:2] = [0.4, -0.7]
    mj.mj_fwdPosition(model, data)
    target = np.eye(4)
    target[:3, :3] = data.xmat[model.body("hand").id].reshape(3, 3)
    target[:3, 3] = data.xpos[model.body("hand").id]
    before = data.qpos.copy()
    args = (
        model,
        data,
        "hand",
        ["a", "b"],
        target,
        [[0.3, -0.6], [0, 0]],
        ["upper", "forearm"],
        {
            "position_error_m": 0.002,
            "rotation_error_deg": 2,
            "joint_margin_rad": 0.03,
            "arm_penetration_m": 0.001,
        },
    )
    passed = screen(*args)
    assert passed["selected"]["endpoint_pass"] and not passed["path_validated"]
    model.geom_pos[model.geom("obstacle").id] = target[:3, 3]
    rejected = screen(*args)
    assert not rejected["selected"]["endpoint_pass"]
    assert rejected["selected"]["arm_penetration_m"] > 0.001
    assert np.array_equal(data.qpos, before)
    bad = target.copy()
    bad[0, 0] *= 2
    with pytest.raises(ValueError):
        screen(*args[:4], bad, *args[5:])
