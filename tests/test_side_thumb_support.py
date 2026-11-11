import numpy as np
import pytest
import trimesh
from real2sim2real.controller_actor_training.bearing_support_plan import (
    directional_exterior_target,
    support_surface,
    validate,
)


def test_side_patch_uses_first_exterior_hit_and_returns_world_normals():
    mesh = trimesh.creation.box(extents=[0.1, 0.08, 0.02])
    mesh.apply_translation([0.2, 0.3, 0.05])
    result = directional_exterior_target(mesh, [0.2, 0.3, 0.05], [-1, 0, 0])
    assert result["accepted"]
    np.testing.assert_allclose(result["targets_m"], [[0.15, 0.3, 0.05]], atol=1e-8)
    np.testing.assert_allclose(result["outward_normals"], [[-1, 0, 0]], atol=1e-8)
    points = np.asarray(result["patch_points_m"])
    assert points.shape == (1, 5, 3)
    assert result["independent_observation_validated"] is False


def test_directional_stencil_rejects_thin_side_face_and_nonunit_directions():
    mesh = trimesh.creation.box(extents=[0.1, 0.08, 0.006])
    assert not directional_exterior_target(mesh, [0, 0, 0], [-1, 0, 0])["accepted"]
    with pytest.raises(ValueError, match="unit"):
        directional_exterior_target(mesh, [0, 0, 0], [-2, 0, 0])


def test_side_thumb_and_four_underside_contacts_keep_anatomical_order():
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.02])
    mesh.apply_translation([0, 0, 0.05])
    p = dict(
        xy_m=[[0, 0], [0, 0.03], [0, 0.01], [0, -0.01], [0, -0.03]],
        thumb_side_contact=dict(anchor_m=[0, 0, 0.05], outward_world=[-1, 0, 0]),
    )
    result = support_surface(mesh, p)
    assert result["accepted"]
    np.testing.assert_allclose(result["targets_m"][0], [-0.05, 0, 0.05])
    np.testing.assert_allclose(
        np.array(result["targets_m"])[1:, 1], [0.03, 0.01, -0.01, -0.03]
    )
    np.testing.assert_allclose(np.array(result["targets_m"])[1:, 2], 0.04)
    np.testing.assert_allclose(result["outward_normals"][0], [-1, 0, 0])
    np.testing.assert_allclose(
        np.array(result["outward_normals"])[1:], [[0, 0, -1]] * 4
    )


def test_side_contact_and_opening_search_cannot_expand_declared_bounds():
    proposal = dict(
        xy_m=[[0, 0]] * 5,
        yaw_rad=0,
        thumb_side_contact=dict(anchor_m=[0, 0, 0.05], outward_world=[-1, 0, 0]),
    )
    config = dict(
        proposals=[proposal],
        maximum_evaluations=100,
        maximum_normal_force_n=10,
        deadline_s=900,
    )
    validate(config)
    with pytest.raises(ValueError, match="gap"):
        validate({**config, "proposals": [{**proposal, "opening_gap_m": 0.021}]})
    with pytest.raises(ValueError, match="horizontal"):
        validate(
            {
                **config,
                "proposals": [
                    {
                        **proposal,
                        "thumb_side_contact": dict(
                            anchor_m=[0, 0, 0], outward_world=[0, 0, 1]
                        ),
                    }
                ],
            }
        )


def test_planning_preload_can_reduce_penetration_without_relaxing_patch_limits():
    proposal = dict(xy_m=[[0, 0]] * 5, yaw_rad=0, preload_m=0.00025)
    config = dict(
        proposals=[proposal],
        maximum_evaluations=100,
        maximum_normal_force_n=10,
        deadline_s=900,
    )
    validate(config)
    for wrong in [-0.0001, 0.0006]:
        with pytest.raises(ValueError, match="preload"):
            validate({**config, "proposals": [{**proposal, "preload_m": wrong}]})


def test_root_frame_constant_rebuild_preserves_retained_hand_object_and_clock():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        set_root_frame,
    )

    m = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="object" pos="0 0 .1"><freejoint/><geom size=".01"/></body>
    <body name="primary"><joint name="finger" axis="0 1 0"/><geom size=".01"/></body>
    <body name="support"><joint name="s_x" axis="1 0 0" type="slide"/>
    <joint name="s_y" axis="0 1 0" type="slide"/>
    <joint name="s_z" axis="0 0 1" type="slide"/><geom size=".01"/></body>
    </worldbody><actuator><motor joint="finger"/></actuator></mujoco>""")
    d = mj.MjData(m)
    d.qpos[2] = 0.12
    d.qpos[7] = 0.8
    d.qpos[8:] = [0.01, 0.02, 0.03]
    d.qvel[:] = np.arange(m.nv) * 0.01
    d.ctrl[0] = 0.3
    d.time = 2.75
    originals = [v.copy() for v in (d.qpos, d.qvel, d.ctrl)]
    set_root_frame(
        m, d, "support", "s_", [0.1, 0.2, 0.3], np.eye(3), preserve_state=True
    )
    for after, before in zip((d.qpos, d.qvel, d.ctrl), originals):
        np.testing.assert_array_equal(after, before)
    assert d.time == 2.75
    np.testing.assert_allclose(m.body_pos[m.body("support").id], [0.1, 0.2, 0.3])
