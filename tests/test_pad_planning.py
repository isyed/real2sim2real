import numpy as np
import pytest
import trimesh

from real2sim2real.retargeting.mesh_microholes import close_microholes
from real2sim2real.retargeting.pad_contacts import gravity_wrench_screen
from real2sim2real.controller_actor_training.pad_grasp_planning import closing_target


def test_microhole_repair_preserves_vertices_and_rejects_large_missing_surface():
    for size in (0.0004, 0.04):
        box = trimesh.creation.box(extents=[size] * 3)
        keep = box.face_normals[:, 0] < 0.9
        box.update_faces(keep)
        vertices, faces = box.vertices.copy(), box.faces.copy()
        if size > 0.001:
            with pytest.raises(ValueError, match="metric repair"):
                close_microholes(box)
        else:
            repaired, report = close_microholes(box)
            assert repaired.is_watertight and repaired.is_winding_consistent
            assert report["faces_added"] == 2
            np.testing.assert_array_equal(repaired.vertices, vertices)
            np.testing.assert_array_equal(repaired.faces[: len(faces)], faces)
            with pytest.raises(ValueError, match="budget"):
                close_microholes(box, max_edges=3)
        np.testing.assert_array_equal(box.faces, faces)


def test_static_wrench_screen_requires_friction_and_balancing_moment():
    p = [[-0.02, 0, 0], [0.02, 0, 0]]
    n = [[-1, 0, 0], [1, 0, 0]]
    good = gravity_wrench_screen(p, n, [0, 0, 0], 0.2, 0.8, 10)
    assert good["feasible"]
    assert not good["actuator_feasibility_validated"]
    assert not gravity_wrench_screen(p, n, [0, 0, 0], 0.2, 0, 10)["feasible"]
    assert not gravity_wrench_screen(p, n, [0, 1, 0], 0.2, 0.8, 10)["feasible"]
    assert not gravity_wrench_screen(p, n, [0, 0, 0], 20, 0.8, 10)["feasible"]
    with pytest.raises(ValueError):
        gravity_wrench_screen(p, n, [0, 0, 0], float("nan"), 0.8, 10)


def test_planned_closure_cannot_acquire_legacy_offsets_or_feedback():
    candidate = dict(pad_plan={}, q_closed=[0.2, 0.3])
    config = dict(rollout=dict(close_offset_rad=0))
    np.testing.assert_equal(closing_target(config, candidate, [0, 0]), [0.2, 0.3])
    for extra in (
        dict(rollout=dict(close_offset_rad=0.1)),
        dict(contact_feedback={"gain": 1}),
        dict(finger_closure={"offsets_rad": {"finger": 0.1}}),
    ):
        with pytest.raises(ValueError, match="cannot be altered"):
            closing_target(dict(config, **extra), candidate, [0, 0])
    np.testing.assert_equal(closing_target(config, {}, [0.5]), [0.5])


def test_patch_force_budget_is_shared_and_area_can_balance_a_moment():
    points = [[x, y, 0] for x in [-0.02, 0.02] for y in [-0.004, 0, 0.004]]
    normals = [[x, 0, 0] for x in [-1, 1] for _ in range(3)]
    labels = [0, 0, 0, 1, 1, 1]
    assert gravity_wrench_screen(
        points, normals, [0, 0.003, 0], 0.5, 0.8, 5, contact_groups=labels
    )["feasible"]
    assert not gravity_wrench_screen(
        [[-0.02, 0, 0], [0.02, 0, 0]],
        [[-1, 0, 0], [1, 0, 0]],
        [0, 0.003, 0],
        0.5,
        0.8,
        5,
    )["feasible"]
    assert not gravity_wrench_screen(
        points, normals, [0, 0, 0], 1.5, 0.8, 5, contact_groups=labels
    )["feasible"]
    assert gravity_wrench_screen(points, normals, [0, 0, 0], 1.5, 0.8, 5)["feasible"]
    with pytest.raises(ValueError, match="groups"):
        gravity_wrench_screen(
            points, normals, [0, 0, 0], 0.5, 0.8, 5, contact_groups=[-1] * 6
        )


def test_depth_review_keeps_thin_foreground_visible_independent_of_face_order():
    from real2sim2real.controller_actor_training.object_grasp_review import panel

    triangles = np.array(
        [
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
            [[-0.1, -0.1, 0.001], [0.1, -0.1, 0.001], [0, 0.1, 0.001]],
        ]
    )
    colors = np.array([[240, 0, 0], [0, 0, 240]])
    a = np.asarray(panel(triangles, colors, 0, 90, np.zeros(3), 100, depth_buffer=True))
    b = np.asarray(
        panel(triangles[::-1], colors[::-1], 0, 90, np.zeros(3), 100, depth_buffer=True)
    )
    np.testing.assert_array_equal(a, b)
    assert a[235, 260, 2] > a[235, 260, 0]


def fixture():
    mujoco = pytest.importorskip("mujoco")
    mesh = trimesh.creation.box(extents=[0.02, 0.012, 0.03])
    vertices = " ".join(map(str, mesh.vertices.ravel()))
    faces = " ".join(map(str, mesh.faces.ravel()))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
    <asset><mesh name="pad_mesh" vertex="{vertices}" face="{faces}"/></asset>
    <worldbody><body name="hand" pos="0 0 .3"><joint name="fixture_x" type="slide" axis="1 0 0" range="-.2 .2"/>
    <geom type="sphere" size=".001" mass=".1" contype="0" conaffinity="0"/>
    <body name="finger" pos=".03 0 0"><joint name="finger" axis="0 0 1" range="-1 1"/>
    <geom name="pad" type="mesh" mesh="pad_mesh" mass=".1"/></body></body>
    <body name="task_object" pos="1 0 .3"><freejoint/><geom name="object_geom" type="sphere" size=".02" mass=".2"/></body>
    </worldbody></mujoco>''')
    return mujoco, model, mujoco.MjData(model)


def test_compiled_pad_frame_and_analytic_position_normal_jacobians():
    from real2sim2real.retargeting.pad_contacts import pad_frame, observe_frames
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    mujoco, model, data = fixture()
    frame = pad_frame(model, "hand", "pad", [0, 1, 0])
    np.testing.assert_allclose(frame["point_body_m"], [0, 0.006, 0], atol=1e-8)
    np.testing.assert_allclose(frame["normal_body"], [0, 1, 0], atol=1e-8)
    with pytest.raises(ValueError, match="owned"):
        pad_frame(model, "hand", "object_geom", [0, 1, 0])
    idx = robot_indices(model, "hand")
    data.qpos[1] = 0.4
    mujoco.mj_fwdPosition(model, data)
    p, n, jp, jn = observe_frames(model, data, [frame], idx, jacobian=True)
    for i in range(2):
        data.qpos[i] += 1e-6
        mujoco.mj_fwdPosition(model, data)
        p2, n2 = observe_frames(model, data, [frame], idx)
        np.testing.assert_allclose((p2 - p) / 1e-6, jp[:, :, i], atol=1e-6)
        np.testing.assert_allclose((n2 - n) / 1e-6, jn[:, :, i], atol=1e-6)
        data.qpos[i] -= 1e-6


def test_known_geometry_pad_fit_preserves_object_and_caller_state():
    from real2sim2real.retargeting.pad_contacts import (
        pad_frame,
        observe_frames,
        fit_pad_contacts,
    )
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    mujoco, model, data = fixture()
    idx = robot_indices(model, "hand")
    frame = pad_frame(model, "hand", "pad", [0, 1, 0])
    data.qpos[:2] = [0.02, 0.5]
    mujoco.mj_fwdPosition(model, data)
    p, n = observe_frames(model, data, [frame], idx)
    before = data.qpos.copy()
    result = fit_pad_contacts(model, data, "hand", [frame], p, n, [0, 0])
    assert max(result["site_errors_m"]) < 1e-5
    assert max(result["normal_errors_deg"]) < 0.02
    np.testing.assert_allclose(result["q"], [0.02, 0.5], atol=0.0001)
    np.testing.assert_array_equal(data.qpos, before)
    from real2sim2real.controller_actor_training.pad_grasp_planning import (
        pad_closure_geometry,
    )

    candidate = dict(pad_plan=dict(frames=[frame]), q_grasp=before[:2], targets_m=p)
    audit = pad_closure_geometry(model, data, idx, candidate, before[:2])
    assert audit["contact_reference"] == "collision_pad"
    assert max(audit["planned_target_error_m"]) < 1e-9
    np.testing.assert_array_equal(data.qpos, before)
    with pytest.raises(ValueError, match="narrow"):
        fit_pad_contacts(
            model,
            data,
            "hand",
            [frame],
            p,
            n,
            [0, 0],
            joint_bounds={"fixture_x": [-0.2, 0.2]},
        )


def test_patch_samples_require_real_pad_and_nearby_opposed_object_surface():
    from real2sim2real.retargeting.pad_contacts import pad_frame, contact_patch_samples

    mujoco, model, data = fixture()
    mujoco.mj_fwdPosition(model, data)
    frame = pad_frame(model, "hand", "pad", [0, 1, 0])
    box = trimesh.creation.box(extents=[0.03, 0.04, 0.04])
    box.apply_translation([0.03, 0.026, 0.3])
    result = contact_patch_samples(model, data, [frame], box)
    assert result["samples_per_pad"] == [5]
    assert result["all_pads_observed"]
    np.testing.assert_allclose(np.asarray(result["points_m"])[:, 1], 0.006, atol=1e-7)
    box.apply_translation([0, 0.003, 0])
    result = contact_patch_samples(model, data, [frame], box)
    assert result["samples_per_pad"] == [0]
    assert not result["all_pads_observed"]


def test_path_audit_detects_intermediate_collision_and_joint_limit_violation():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.controller_actor_training.pad_grasp_planning import path_audit

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom type="sphere" size=".01"/><body name="hand"><joint type="slide" axis="1 0 0" range="-.1 .1"/>
    <geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    idx = robot_indices(model, "hand")
    report = path_audit(model, data, idx, [[-0.05], [0.05]])
    assert report["max_penetration_m"] > 0.019
    assert report["within_joint_limits"]
    assert report["samples"] >= 51
    assert not path_audit(model, data, idx, [[-0.2], [-0.1]])["within_joint_limits"]
    assert data.qpos[0] == 0
