from copy import deepcopy
import json

import numpy as np
import pytest
import trimesh

from real2sim2real.retargeting.object_grasp import cue_cost, surface_pairs
from real2sim2real.controller_actor_training.object_grasp_pilot import (
    target_at,
    clearance_feedback,
    load_plan,
    verify_selected_trial,
)


def test_geometry_proposals_respect_width_and_table_clearance_without_hands():
    mesh = trimesh.creation.box(extents=[0.04, 0.2, 0.04])
    mesh.apply_translation([0, 0, 0.06])
    proposals = surface_pairs(
        mesh,
        min_width_m=0.02,
        max_width_m=0.06,
        minimum_height_m=0.045,
        max_candidates=4,
    )
    assert proposals
    for proposal in proposals:
        assert 0.02 <= proposal["width_m"] <= 0.06
        assert min(np.asarray(proposal["targets_m"])[:, 2]) >= 0.045
        np.testing.assert_allclose(np.abs(proposal["close_axis"]), [1, 0, 0], atol=1e-8)
    assert not surface_pairs(
        mesh,
        min_width_m=0.045,
        max_width_m=0.06,
        minimum_height_m=0.045,
        max_candidates=4,
    )


def test_uncertain_cues_do_not_turn_wide_objects_into_narrow_grasps():
    mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.1])
    mesh.apply_translation([0, 0, 0.06])
    assert not surface_pairs(
        mesh,
        min_width_m=0.02,
        max_width_m=0.06,
        minimum_height_m=0.04,
        cues=[dict(position_m=[0, 0, 0.05], confidence=1, weight=100)],
    )
    assert cue_cost([0, 0, 0], [dict(position_m=[1, 0, 0], confidence=0)]) == 0
    assert (
        cue_cost([0, 0, 0], [dict(position_m=[1, 0, 0], confidence=1, weight=100)])
        == 0.02
    )


def test_grasp_schedule_keeps_lift_separate_from_closure_and_holds_height():
    pre = np.array([0.1, 0.0, 0.3])
    opened = np.array([0.1, 0.0, 0.0])
    closed = np.array([0.7, 0.5, 0.0])
    settings = dict(approach_s=1.0, close_s=1.0, lift_s=1.0, lift_m=0.07)
    np.testing.assert_equal(target_at(0, pre, opened, closed, 2, settings)[0], pre)
    np.testing.assert_equal(target_at(1, pre, opened, closed, 2, settings)[0], opened)
    np.testing.assert_equal(target_at(2, pre, opened, closed, 2, settings)[0], closed)
    for t in [3.0, 4.0, 5.0]:
        target, phase = target_at(t, pre, opened, closed, 2, settings)
        np.testing.assert_allclose(target, [0.7, 0.5, 0.07])
        assert phase == "hold"


def test_contact_penetration_jacobian_and_robot_joint_ownership():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        contact_penetrations,
    )

    xml = """<mujoco><worldbody><geom type="sphere" size=".05"/><body name="hand" pos=".08 0 0"><joint name="slide" type="slide" axis="1 0 0" range="-.2 .2"/><geom type="sphere" size=".05" mass="1"/></body><body name="object" pos="0 0 .5"><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    idx = robot_indices(model, "hand")
    assert idx["q"].tolist() == [0]
    mujoco.mj_forward(model, data)
    value, jac = contact_penetrations(model, data, idx, jacobian=True)
    assert value.max() > 0.019
    data.qpos[0] += 1e-6
    mujoco.mj_forward(model, data)
    shifted = contact_penetrations(model, data, idx)
    np.testing.assert_allclose((shifted - value) / 1e-6, jac[:, 0], atol=1e-6)


def test_feedback_uses_object_clearance_and_respects_speed_and_travel():
    settings = dict(
        gain_s_inv=3.0,
        max_raise_speed_m_s=0.1,
        max_lower_speed_m_s=0.02,
        max_wrist_lift_m=0.18,
    )
    assert clearance_feedback(0.02, 0, 0.07, 0.1, settings) == pytest.approx(0.03)
    assert clearance_feedback(0.18, 0, 0.07, 0.1, settings) == pytest.approx(0.18)
    assert clearance_feedback(0.10, 0.09, 0.07, 0.1, settings) == pytest.approx(0.098)
    assert clearance_feedback(0, 0.09, 0.07, 0.1, settings) == 0


def test_saved_plan_rejects_changed_scene_or_settled_task(tmp_path):
    from real2sim2real.artifacts import digest

    (tmp_path / "scene").mkdir()
    (tmp_path / "planning").mkdir()
    receipt = tmp_path / "scene/receipt.json"
    receipt.write_text('{"scene":"first"}')
    settled = dict(accepted=True, object_qpos=[0, 0, 0, 1, 0, 0, 0])
    (tmp_path / "settling.json").write_text(json.dumps(settled))
    plan = dict(scene_receipt_sha256=digest(receipt), settling=settled)
    (tmp_path / "planning/plan.json").write_text(json.dumps(plan))
    config = dict(output=str(tmp_path))
    assert load_plan(config) == plan
    changed = deepcopy(settled)
    changed["object_qpos"][0] = 0.01
    (tmp_path / "settling.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="initialization"):
        load_plan(config)
    (tmp_path / "settling.json").write_text(json.dumps(settled))
    receipt.write_text('{"scene":"second"}')
    with pytest.raises(ValueError, match="another scene"):
        load_plan(config)


def test_frozen_selection_cannot_change_grasp_or_recorded_outcome():
    candidate = dict(candidate_id=3, accepted=True, q_grasp=[0.2, 0.4])
    plan = dict(candidates=[candidate])
    summary = dict(success=True, candidate_id=3, states_sha256="retained-states")
    selected = dict(candidate=deepcopy(candidate), **summary)
    evaluation = dict(
        selected_trial="chosen", trials=[dict(directory="chosen", **summary)]
    )
    verify_selected_trial(evaluation, selected, plan)
    changed = deepcopy(selected)
    changed["states_sha256"] = "different-states"
    with pytest.raises(ValueError, match="pinned evaluation"):
        verify_selected_trial(evaluation, changed, plan)
    changed = deepcopy(selected)
    changed["candidate"]["q_grasp"][0] = 0.7
    with pytest.raises(ValueError, match="accepted contact plan"):
        verify_selected_trial(evaluation, changed, plan)


def test_candidate_height_and_cue_validation_fail_closed():
    mesh = trimesh.creation.box(extents=[0.04, 0.2, 0.04])
    mesh.apply_translation([0, 0, 0.06])
    assert not surface_pairs(
        mesh, min_width_m=0.02, max_width_m=0.06, minimum_height_m=0.10
    )
    with pytest.raises(ValueError):
        cue_cost([0, 0, 0], [dict(position_m=[1, 0, 0], confidence=1, weight=-1)])
    with pytest.raises(ValueError):
        surface_pairs(
            mesh,
            min_width_m=0.02,
            max_width_m=0.06,
            minimum_height_m=0.04,
            finger_offsets_m=[float("nan")],
        )


def test_five_finger_targets_are_distinct_and_follow_declared_spacing():
    mesh = trimesh.creation.box(extents=[0.04, 0.2, 0.04])
    mesh.apply_translation([0, 0, 0.06])
    row = surface_pairs(
        mesh,
        min_width_m=0.02,
        max_width_m=0.06,
        minimum_height_m=0.045,
        max_candidates=1,
        finger_offsets_m=[0.03, 0.01, -0.01, -0.03],
    )[0]
    target = np.asarray(row["targets_m"])
    assert target.shape == (5, 3)
    distances = (target[1:] - np.asarray(row["center_m"])) @ np.asarray(
        row["along_axis"]
    )
    np.testing.assert_allclose(distances, [0.03, 0.01, -0.01, -0.03], atol=1e-8)


@pytest.mark.parametrize("extra", ["gravity", "weld", "actuator"])
def test_object_assistance_is_rejected(extra):
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_unassisted_model,
    )

    option = '<option gravity="0 0 0"/>' if extra == "gravity" else ""
    joint = '<freejoint name="object_joint"/>'
    equality = (
        '<equality><weld body1="task_object"/></equality>' if extra == "weld" else ""
    )
    actuator = (
        '<actuator><motor joint="object_joint"/></actuator>'
        if extra == "actuator"
        else ""
    )
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco>{option}<worldbody><body name="task_object">{joint}<geom type="sphere" size=".05" mass="1"/></body></worldbody>{equality}{actuator}</mujoco>'
    )
    with pytest.raises(ValueError):
        verify_unassisted_model(model)
