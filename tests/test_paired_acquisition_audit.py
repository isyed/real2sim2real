import json

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.paired_acquisition_audit import (
    paired_waypoints,
    pinned_json,
)


def hand(name, pre, opened, closed):
    return dict(
        joint_names=[name], q_pregrasp=[pre], q_open=[opened], q_closed=[closed]
    )


def test_paired_path_keeps_each_native_joint_and_all_three_states():
    result = paired_waypoints([hand("left", -0.2, 0, 0.1), hand("right", 0.2, 0.1, 0)])
    np.testing.assert_array_equal(result, [[-0.2, 0.2], [0, 0.1], [0.1, 0]])
    with pytest.raises(ValueError, match="distinct"):
        paired_waypoints([hand("left", 0, 0, 0), hand("left", 0, 0, 0)])
    with pytest.raises(ValueError, match="finite"):
        paired_waypoints([hand("left", 0, 0, np.nan), hand("right", 0, 0, 0)])


def test_changed_parent_cannot_supply_a_new_admission(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"success": False}))
    receipt = dict(path=str(path), sha256=digest(path))
    assert pinned_json(receipt)["success"] is False
    path.write_text(json.dumps({"success": True}))
    with pytest.raises(ValueError, match="changed"):
        pinned_json(receipt)


def test_simultaneous_crossing_is_missed_by_held_primary_check():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.controller_actor_training.pad_grasp_planning import path_audit

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="primary"><joint name="x" type="slide" axis="1 0 0" range="-.3 .3"/>
        <geom name="primary_pad" type="sphere" size=".02" mass="1"/></body>
      <body name="support"><joint name="y" type="slide" axis="0 1 0" range="-.3 .3"/>
        <geom name="support_pad" type="sphere" size=".02" mass="1"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    first, second = [robot_indices(model, h) for h in ("primary", "support")]
    data.qpos[first["q"]] = 0.2
    data.qpos[second["q"]] = -0.2
    before = data.qpos.copy()
    held = path_audit(model, data, second, [[-0.2], [0.2]])
    combined = {
        k: np.concatenate([first[k], second[k]])
        for k in ("q", "joints", "dofs", "geoms")
    }
    paired = path_audit(model, data, combined, [[-0.2, -0.2], [0.2, 0.2]])
    assert held["max_penetration_m"] == 0
    assert paired["max_penetration_m"] == pytest.approx(0.04)
    assert paired["worst_contact"]["fraction"] == pytest.approx(0.5)
    np.testing.assert_array_equal(data.qpos, before)


def test_visibility_rejects_hidden_back_surface_without_depth_fabrication():
    import trimesh
    from real2sim2real.egocentric.contact_observations import visible_surface

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    camera = np.eye(4)
    camera[:3, 3] = [0, 0, 2]
    # Even a deliberately front-facing normal cannot make the occluded back
    # surface visible. This is a geometric screen, not an observation.
    result = visible_surface(
        mesh,
        np.array([[0, 0, 0.5], [0, 0, -0.5]]),
        np.array([[0, 0, 1], [0, 0, 1]]),
        camera,
        0.002,
    )
    assert result.tolist() == [True, False]
