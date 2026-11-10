import numpy as np
import pytest
import trimesh

from real2sim2real.controller_actor_training.bilateral_grasp import wall_targets
from real2sim2real.controller_actor_training.support_control import simultaneous_target


def test_rim_targets_distinguish_inner_thumb_from_outer_fingers_on_real_surface():
    mesh = trimesh.creation.annulus(r_min=0.08, r_max=0.09, height=0.1, sections=64)
    offsets = [-0.02, -0.01, 0, 0.01, 0.02]
    cup = wall_targets(mesh, [0, 0], [1, 0, 0], offsets, [0.02] * 5, "body_cup")
    rim = wall_targets(mesh, [0, 0], [1, 0, 0], offsets, [0.02] * 5, "rim_pinch")
    assert cup is not None and rim is not None
    p, n = rim
    assert n[0, 0] < -0.9 and (n[1:, 0] > 0.9).all()
    assert np.linalg.norm(p[0, :2]) == pytest.approx(0.08, abs=0.0002)
    assert (np.linalg.norm(p[1:, :2], axis=1) > 0.089).all()
    assert cup[0][0, 0] > rim[0][0, 0]
    assert wall_targets(mesh, [0, 0], [1, 0, 0], offsets, [0.2] * 5, "body_cup") is None
    with pytest.raises(ValueError):
        wall_targets(mesh, [0, 0], [0, 0, 1], offsets, [0.02] * 5, "body_cup")


def test_two_hand_schedule_closes_before_lift_and_shares_feedback_displacement():
    rollout = dict(approach_s=1.0, close_s=0.8, lift_s=2.0, lift_m=0.07)
    pre = np.array([0.1, 0, 0.02, 0.0])
    opened = np.array([0, 0, 0.02, 0.1])
    closed = np.array([0, 0, 0.02, 0.6])
    originals = [v.copy() for v in (pre, opened, closed)]
    for time in [0.0, 0.5, 1.0, 1.4, 1.79]:
        q, phase = simultaneous_target(
            time, pre, opened, closed, [0, 1, 2], 0.03, 0.03, rollout
        )
        assert phase in ["approach", "close"]
        assert q[2] == pytest.approx(0.02)
        assert q[3] <= 0.6
    q, phase = simultaneous_target(
        2.8, pre, opened, closed, [0, 1, 2], 0.075, 0.03, rollout
    )
    assert (
        phase == "lift" and q[2] == pytest.approx(0.065) and q[3] == pytest.approx(0.6)
    )
    q, phase = simultaneous_target(
        4.0, pre, opened, closed, [0, 1, 2], 0.10, 0.03, rollout
    )
    assert phase == "hold" and q[2] == pytest.approx(0.09)
    for actual, before in zip((pre, opened, closed), originals):
        np.testing.assert_array_equal(actual, before)


def test_support_fixture_uses_identical_pad_ik_envelope_for_another_namespace():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.pad_contacts import pad_frame, fit_pad_contacts

    box = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    vertices = " ".join(map(str, box.vertices.ravel()))
    faces = " ".join(map(str, box.faces.ravel()))
    m = mj.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="mesh" vertex="{vertices}" face="{faces}"/></asset>
    <worldbody><body name="hand"><joint name="support_x" type="slide" axis="1 0 0" range="-.3 .3"/>
    <geom name="pad" type="mesh" mesh="mesh"/></body></worldbody></mujoco>''')
    d = mj.MjData(m)
    frame = pad_frame(m, "hand", "pad", [0, 1, 0])
    with pytest.raises(ValueError, match="narrow"):
        fit_pad_contacts(
            m,
            d,
            "hand",
            [frame],
            [[0.1, 0.005, 0]],
            [[0, 1, 0]],
            [0],
            fixture_prefix="support_",
            joint_bounds={"support_x": [-0.1, 0.1]},
        )
    fit = fit_pad_contacts(
        m,
        d,
        "hand",
        [frame],
        [[0.1, 0.005, 0]],
        [[0, 1, 0]],
        [0],
        fixture_prefix="support_",
        position_scale_m=0.001,
    )
    assert fit["q"][0] <= 0.06
    np.testing.assert_array_equal(d.qpos, [0.0])


def test_closed_wrist_rebase_retains_waypoints_and_frees_fixture_margin():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.bilateral_grasp import (
        rebase_closed_wrist,
    )
    from scipy.spatial.transform import Rotation

    m = mj.MjModel.from_xml_string("""<mujoco><worldbody><body name="object" pos="1 0 1"><freejoint/><geom size=".02"/></body>
    <body name="hand"><joint name="f_x" axis="1 0 0" type="slide" range="-1 1"/>
    <joint name="f_y" axis="0 1 0" type="slide" range="-1 1"/><joint name="f_z" axis="0 0 1" type="slide" range="-1 1"/>
    <joint name="f_rx" axis="1 0 0" range="-1 1"/><joint name="f_ry" axis="0 1 0" range="-1 1"/>
    <joint name="f_rz" axis="0 0 1" range="-1 1"/><geom size=".01"/>
    <body pos=".1 0 0"><joint name="finger" axis="0 1 0"/><geom size=".01"/></body></body></worldbody></mujoco>""")
    d = mj.MjData(m)
    d.qpos[0] = 1.2
    h = dict(
        root_body="hand",
        fixture_prefix="f_",
        joint_names=["f_" + a for a in ["x", "y", "z", "rx", "ry", "rz"]] + ["finger"],
        base_position_m=[0.2, 0.1, 0.3],
        base_rotation=Rotation.from_euler("xyz", [0.2, 0.3, 0.1]).as_matrix().tolist(),
    )
    for i, key in enumerate(["q_pregrasp", "q_open", "q_grasp", "q_closed"]):
        h[key] = [0.05, 0.02, 0.1 + 0.01 * i, 0.2, 0.3, 0.25, 0.4]
    new = rebase_closed_wrist(m, d, h)
    assert new["rebase_maximum_pose_residual"] < 1e-10
    np.testing.assert_allclose(new["q_closed"][:6], np.zeros(6), atol=1e-10)
    assert d.qpos[0] == pytest.approx(1.2)
    assert new["q_pregrasp"][6] == pytest.approx(0.4)


def test_joint_two_hand_pad_fit_includes_mutual_contacts_and_preserves_object_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.pad_contacts import pad_frame, fit_pad_contacts

    box = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    vertices = " ".join(map(str, box.vertices.ravel()))
    faces = " ".join(map(str, box.faces.ravel()))
    m = mj.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="mesh" vertex="{vertices}" face="{faces}"/></asset><worldbody>
    <body name="left" pos="-.03 0 .2"><joint name="l_x" type="slide" axis="1 0 0" range="-.2 .2"/><geom name="l_pad" type="mesh" mesh="mesh"/></body>
    <body name="right" pos=".03 0 .2"><joint name="r_x" type="slide" axis="1 0 0" range="-.2 .2"/><geom name="r_pad" type="mesh" mesh="mesh"/></body>
    <body pos="1 0 1"><freejoint/><geom size=".01"/></body></worldbody></mujoco>''')
    d = mj.MjData(m)
    before = d.qpos.copy()
    frames = [
        pad_frame(m, "left", "l_pad", [1, 0, 0]),
        pad_frame(m, "right", "r_pad", [-1, 0, 0]),
    ]
    fit = fit_pad_contacts(
        m,
        d,
        ["left", "right"],
        frames,
        [[-0.002, 0, 0.2], [0.002, 0, 0.2]],
        [[1, 0, 0], [-1, 0, 0]],
        [0.03, -0.03],
        fixture_prefix=["l_", "r_"],
        position_scale_m=0.001,
    )
    assert max(fit["site_errors_m"]) < 1e-5
    assert fit["penetration_m"].max(initial=0) < 1e-5
    np.testing.assert_allclose(fit["q"], [0.023, -0.023], atol=1e-5)
    np.testing.assert_array_equal(d.qpos, before)
    with pytest.raises(ValueError, match="distinct"):
        fit_pad_contacts(
            m,
            d,
            ["left", "left"],
            frames,
            [[-0.002, 0, 0.2], [0.002, 0, 0.2]],
            [[1, 0, 0], [-1, 0, 0]],
            [0.03, -0.03],
            fixture_prefix=["l_", "l_"],
        )


def test_stationary_hold_cannot_promote_demonstrated_motion_and_requires_completion():
    from real2sim2real.controller_actor_training.bilateral_evaluate import (
        assess_motion_condition,
    )

    result = dict(
        success=True, simulation_final_time_s=10.0, simulation_requested_duration_s=10.0
    )
    score = assess_motion_condition(
        result, "stationary_diagnostic", dict(position_m=0.02, rotation_deg=10)
    )
    assert score["diagnostic_hold_pass"] and score["complete"]
    assert score["combined_pass"] is False and score["reference_pass"] is None
    result["simulation_final_time_s"] = 5.0
    assert not assess_motion_condition(result, "stationary_diagnostic", {})[
        "diagnostic_hold_pass"
    ]
    with pytest.raises(ValueError):
        assess_motion_condition(result, "unknown", {})
