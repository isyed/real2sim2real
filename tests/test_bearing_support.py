import numpy as np
import pytest
import trimesh

from real2sim2real.controller_actor_training.bearing_support_plan import (
    exterior_targets,
    validate,
)
from real2sim2real.retargeting.pad_contacts import gravity_wrench_screen


def test_exterior_rays_select_first_lower_surface_not_cavity_floor():
    # Two separated layers: first intersection from below must be the lower one.
    lower = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    lower.apply_translation([0, 0, 0.015])
    upper = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    upper.apply_translation([0, 0, 0.055])
    mesh = trimesh.util.concatenate([lower, upper])
    result = exterior_targets(mesh, [[0, 0], [0, 0]], [1, -1])
    assert result["accepted"]
    np.testing.assert_allclose(np.asarray(result["targets_m"])[:, 2], [0.06, 0.01])
    assert result["outward_normals"][0][2] == 1
    assert result["outward_normals"][1][2] == -1
    assert result["independent_observation_validated"] is False


def test_center_hit_does_not_certify_finite_patch_at_a_thin_edge():
    mesh = trimesh.creation.box(extents=[0.006, 0.1, 0.01])
    result = exterior_targets(mesh, [[0, 0]], [-1])
    assert not result["accepted"]
    assert result["rejected_for"] == ["missing_exterior_patch"]


def test_underside_normal_must_face_exterior_and_cliff_is_rejected():
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces[:, ::-1], process=False
    )
    assert exterior_targets(mesh, [[0, 0]], [-1])["rejected_for"] == [
        "wrong_facing_or_steep_exterior"
    ]
    left = trimesh.creation.box(extents=[0.05, 0.1, 0.01])
    left.apply_translation([-0.025, 0, 0.02])
    right = trimesh.creation.box(extents=[0.05, 0.1, 0.01])
    right.apply_translation([0.025, 0, 0.04])
    mesh = trimesh.util.concatenate([left, right])
    assert exterior_targets(mesh, [[-0.001, 0]], [-1])["rejected_for"] == [
        "discontinuous_exterior_patch"
    ]


def test_contact_forces_balance_weight_and_torque_not_just_normal_pressure():
    p = np.array([[-0.04, 0, 0.1], [0.04, 0, 0.1]])
    result = gravity_wrench_screen(p, [[0, 0, -1]] * 2, [0, 0, 0.1], 0.5, 0, 4)
    assert result["feasible"]
    forces = np.asarray(result["contact_forces_world_n"])
    np.testing.assert_allclose(forces.sum(0), [0, 0, 4.905], atol=1e-8)
    np.testing.assert_allclose(
        np.cross(p - [0, 0, 0.1], forces).sum(0), [0, 0, 0], atol=1e-8
    )
    np.testing.assert_allclose(result["equilibrium_residual"], np.zeros(6), atol=1e-8)
    failed = gravity_wrench_screen(p, [[0, 0, 1]] * 2, [0, 0, 0.1], 0.5, 0, 4)
    assert not failed["feasible"] and failed["contact_forces_world_n"] is None


def test_support_search_is_bounded_before_simulator_import():
    config = dict(
        proposals=[dict(xy_m=[[0, 0]] * 5, yaw_rad=0)],
        maximum_evaluations=100,
        maximum_normal_force_n=4,
        deadline_s=900,
    )
    validate(config)
    with pytest.raises(ValueError, match="twelve"):
        validate({**config, "proposals": config["proposals"] * 13})
    with pytest.raises(ValueError, match="positions"):
        exterior_targets(trimesh.creation.box(), [[np.nan, 0]], [-1])


def test_survey_uses_declared_table_height_and_does_not_admit_control():
    from real2sim2real.controller_actor_training.bearing_support_plan import (
        survey_underside,
    )

    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    mesh.apply_translation([0, 0, 0.05])
    valid = survey_underside(mesh, [[0, 0], [0.2, 0]], table_top_z_m=0)
    assert valid["sampled_positions"] == 2 and len(valid["sites"]) == 1
    assert valid["control_admitted"] is False
    assert not survey_underside(mesh, [[0, 0]], table_top_z_m=0.04)["sites"]
    with pytest.raises(ValueError, match="Bounded"):
        survey_underside(mesh, [[0, 0]] * 4097, table_top_z_m=0)


def test_path_rejection_retains_collision_pair_and_location_without_changing_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="obstacle" size=".01" pos=".05 0 0"/>
    <body name="hand"><joint name="x" type="slide" axis="1 0 0" range="-.2 .2"/>
    <geom name="pad" size=".01"/></body></worldbody></mujoco>""")
    data = mj.MjData(model)
    result = path_audit(model, data, robot_indices(model, "hand"), [[0], [0.1]])
    assert result["max_penetration_m"] == pytest.approx(0.02)
    assert set(result["worst_contact"]["geoms"]) == {"pad", "obstacle"}
    assert result["worst_contact"]["fraction"] == pytest.approx(0.5)
    assert result["worst_contact"]["joint_positions"] == pytest.approx([0.05])
    assert data.qpos[0] == 0
