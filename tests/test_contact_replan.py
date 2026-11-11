from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.arm_clearance_audit import (
    geometry_height_bounds,
    table_gap,
)

from real2sim2real.controller_actor_training.contact_replan import (
    propose,
)


def candidate():
    return dict(
        joint_names=["fixture_x", "right_index_PIP"],
        q_pregrasp=[0.0, 0.2],
        q_open=[0.0, 0.3],
        q_grasp=[0.0, 0.5],
        support=dict(
            fixture_prefix="support_",
            joint_names=["support_x", "support_y", "support_z", "support_rx"],
            base_position_m=[0.2, 0.1, 0.03],
            base_rotation=np.eye(3).tolist(),
            q_pregrasp=[0.12, 0.01, -0.004, 0.1],
            q_open=[0.0, 0.01, -0.004, 0.1],
            q_grasp=[0.0, 0.01, 0.0, 0.1],
            targets_m=[[0.0, 0.1, 0.07]],
        ),
    )


def test_support_tilt_preserves_pivot_and_horizontal_insertion_vertical_seating():
    original = candidate()
    before = deepcopy(original)
    pivot = np.array([0.0, 0.1, 0.07])
    rotation = Rotation.from_rotvec([0, -0.2, 0]).as_matrix()
    new = propose(original, [0, -0.2, 0], pivot, {"right_index_PIP": -0.1})
    a, b = original["support"], new["support"]
    root = np.array(a["base_position_m"]) + np.array(a["q_grasp"][:3])
    # An arbitrary contact-local point at the pivot stays exactly at the pivot.
    point_local = pivot - root
    transformed = (
        np.array(b["base_position_m"])
        + np.array(b["q_grasp"][:3])
        + np.array(b["base_rotation"]) @ point_local
    )
    np.testing.assert_allclose(transformed, pivot, atol=1e-12)
    np.testing.assert_allclose(b["base_rotation"], rotation)
    for start, end in [("q_pregrasp", "q_open"), ("q_open", "q_grasp")]:
        np.testing.assert_allclose(
            np.array(b[end]) - b[start], np.array(a[end]) - a[start]
        )
    assert new["q_grasp"][1] == pytest.approx(0.4)
    assert original == before


@pytest.mark.parametrize(
    "rotation,pivot,offsets",
    [
        ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], {}),
        ([0.0, float("nan"), 0.0], [0.0, 0.0, 0.0], {}),
        ([0.0, 0.0, 0.0], [0.0, 0.0], {}),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {"fixture_x": 0.1}),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {"unknown": 0.1}),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {"right_index_PIP": 0.4}),
    ],
)
def test_reject_unbounded_or_nonfinger_changes(rotation, pivot, offsets):
    with pytest.raises(ValueError):
        propose(candidate(), rotation, pivot, offsets)


def test_clearance_uses_all_collision_vertices_and_is_conservative_outside_table():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco>
    <asset><mesh name="cube" vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
    -1 -1 1  1 -1 1  1 1 1  -1 1 1" scale=".01 .02 .03"/></asset>
    <worldbody><geom name="table" type="box" size=".2 .2 .01" pos="0 0 -.01"/>
    <body pos=".4 0 .05" euler="0 90 0"><geom name="hand" type="mesh" mesh="cube"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_fwdPosition(model, data)
    hand, table = model.geom("hand").id, model.geom("table").id
    assert geometry_height_bounds(model, data, hand) == pytest.approx((0.04, 0.06))
    assert table_gap(model, data, [hand], table) == pytest.approx(0.04)
    # Outside the finite table, lowering still fails the conservative plane screen.
    model.body_pos[1, 2] = 0
    mj.mj_fwdPosition(model, data)
    assert table_gap(model, data, [hand], table) == pytest.approx(-0.01)


def test_extra_insertion_gap_leaves_contact_pose_and_lateral_approach_unchanged():
    old = propose(candidate(), [0, -0.2, 0], [0, 0.1, 0.07], {})
    new = propose(candidate(), [0, -0.2, 0], [0, 0.1, 0.07], {}, 0.008)
    assert old["support"]["q_grasp"] == new["support"]["q_grasp"]
    for key in ("q_open", "q_pregrasp"):
        np.testing.assert_allclose(
            np.array(new["support"][key]) - old["support"][key], [0, 0, -0.008, 0]
        )
    with pytest.raises(ValueError):
        propose(candidate(), [0, 0, 0], [0, 0, 0], {}, -0.001)


def test_loaded_wrist_rebase_preserves_all_body_world_poses_and_object_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.loaded_finger_replan import (
        rebase_wrist,
    )
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        set_wrist_frame,
    )
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="object" pos=".1 .2 .3"><freejoint/><geom size=".02"/></body>
    <body name="hand"><joint name="fixture_x" type="slide" axis="1 0 0"/>
    <joint name="fixture_y" type="slide" axis="0 1 0"/>
    <joint name="fixture_z" type="slide" axis="0 0 1"/>
    <joint name="fixture_rx" axis="1 0 0"/><joint name="fixture_ry" axis="0 1 0"/>
    <joint name="fixture_rz" axis="0 0 1"/><geom size=".01"/>
    <body pos=".1 0 0"><joint name="finger" axis="0 1 0"/><geom size=".01"/></body>
    </body></worldbody></mujoco>""")
    data = mj.MjData(model)
    base = {"robot": {"root_body": "hand"}}
    set_wrist_frame(
        model,
        data,
        base,
        [0.2, -0.1, 0.3],
        Rotation.from_euler("xyz", [0.2, 0.3, -0.4]).as_matrix(),
    )
    idx = robot_indices(model, "hand")
    data.qpos[idx["q"]] = [0.1, -0.2, 0.3, 0.2, -0.3, 0.4, 0.5]
    mj.mj_fwdPosition(model, data)
    pos, rot, obj = data.xpos.copy(), data.xmat.copy(), data.qpos[:7].copy()
    rebase_wrist(
        model, data, base, idx, [model.joint(int(j)).name for j in idx["joints"]]
    )
    np.testing.assert_allclose(data.xpos, pos, atol=1e-12)
    np.testing.assert_allclose(data.xmat, rot, atol=1e-12)
    np.testing.assert_array_equal(data.qpos[:7], obj)
    np.testing.assert_array_equal(data.qpos[idx["q"][:6]], np.zeros(6))


def test_lower_object_screen_latches_at_boundary_even_with_previous_phase_label(
    monkeypatch,
):
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training import (
        contact_replan,
        object_grasp_pilot,
    )

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="table" type="box" size="1 1 .01" pos="0 0 -.01"/>
    <body name="task_object" pos=".8 0 .1"><freejoint name="task_object_free"/><geom size=".01"/></body>
    <body name="right"><joint name="fixture_x" type="slide" axis="1 0 0" range="-1 1"/>
    <joint name="fixture_y" type="slide" axis="0 1 0" range="-1 1"/>
    <joint name="fixture_z" type="slide" axis="0 0 1" range="-1 1"/><geom size=".005"/></body>
    <body name="left"><joint name="support_x" type="slide" axis="1 0 0" range="-1 1"/>
    <joint name="support_y" type="slide" axis="0 1 0" range="-1 1"/>
    <joint name="support_z" type="slide" axis="0 0 1" range="-1 1"/><geom size=".005"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    monkeypatch.setattr(
        object_grasp_pilot, "load_scene", lambda config: (model, data, None)
    )
    base = dict(
        robot=dict(root_body="right"),
        support_robot=dict(root_body="left", fixture_prefix="support_"),
        support_controller=dict(approach_s=1.0, seat_s=0.5, max_alignment_m=0.04),
        rollout=dict(approach_s=1.0, close_s=1.0, lift_s=1.0),
        planning=dict(max_penetration_m=0.003),
    )
    ref = np.eye(4)
    ref[:3, 3] = [0.8, 0, 0.1]
    c = dict(
        base_position_m=[-0.5, 0, 0.1],
        base_rotation=np.eye(3),
        q_pregrasp=[0, 0, 0],
        q_open=[0, 0, 0],
        q_grasp=[0, 0, 0],
        support=dict(
            root_body="left",
            fixture_prefix="support_",
            base_position_m=[0, 0, 0.03],
            base_rotation=np.eye(3),
            joint_names=["support_x", "support_y", "support_z"],
            reference_world_from_object=ref,
            q_pregrasp=[0, 0, 0],
            q_open=[0, 0, 0],
            q_grasp=[0, 0, 0],
        ),
    )
    parent = dict(
        records=[
            dict(time_s=t, phase=p)
            for t, p in [
                (2.98, "lift"),
                (3.0, "lift"),
                (3.5, "support_approach"),
                (4.5, "hold"),
            ]
        ]
    )
    states = dict(qpos=np.tile(model.qpos0, (4, 1)))
    result = contact_replan.screen(base, c, parent, states, 0.003)
    assert result["cases"][0]["minimum_support_table_gap_m"] == pytest.approx(0.025)
    assert result["cases"][1]["minimum_support_table_gap_m"] == pytest.approx(0.021)
    assert not result["rejected_for"]
