import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.retargeting.conditional_hand_scene import (
    align_pad_seed,
    prepare,
    sample_paths,
    assess_obstacle_restoration,
)


@pytest.mark.parametrize(
    "failure", ["missing", "obstacle", "isolated", "alignment", "nonfinite", "none"]
)
def test_restoration_gate_requires_every_stage(failure):
    good = dict(maximum_penetration_m=0.001, within_joint_limits=True)
    isolated = good.copy()
    restored = {"nominal": good.copy(), "expanded": good.copy()}
    error = 0.004
    if failure == "missing":
        del restored["expanded"]
    if failure == "obstacle":
        restored["expanded"]["maximum_penetration_m"] = 0.004
    if failure == "isolated":
        isolated["within_joint_limits"] = False
    if failure == "alignment":
        error = 0.009
    if failure == "nonfinite":
        restored["expanded"]["maximum_penetration_m"] = float("nan")
    result = assess_obstacle_restoration(
        isolated,
        restored,
        required_scenarios=["nominal", "expanded"],
        target_error_m=error,
        normal_error_deg=10.0,
        max_target_error_m=0.008,
        max_normal_error_deg=25.0,
        max_penetration_m=0.003,
    )
    assert result["conditional_approach_admitted"] == (failure == "none")
    assert not result["physics_admitted"] and not result["observation_admitted"]


def test_scene_preparation_preserves_existing_output(tmp_path):
    output = tmp_path / "scene.xml"
    output.write_text("frozen comparison")
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "missing-source.xml", "hand", {}, output)
    assert output.read_text() == "frozen comparison"


def test_pad_seed_recovers_proper_rigid_transform():
    p = np.array([[0.0, 0, 0], [0.03, 0, 0], [0, 0.02, 0]])
    n = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0]])
    r = Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_matrix()
    offset = np.array([0.1, -0.1, 0.3])
    x, q = align_pad_seed(p, n, p @ r.T + offset, n @ r.T)
    assert (
        np.allclose(x, offset) and np.allclose(q, r) and np.isclose(np.linalg.det(q), 1)
    )


def test_full_hand_path_reports_palm_table_collision():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    m = mj.MjModel.from_xml_string(
        """<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/><body name="hand"><joint name="fixture_z" type="slide" axis="0 0 1" range="0 1"/><geom name="palm" type="box" size=".02 .02 .02"/><body name="finger" pos=".1 0 .1"><geom name="thumb_elastomer" type="sphere" size=".01"/></body></body></worldbody></mujoco>"""
    )
    d = mj.MjData(m)
    idx = robot_indices(m, "hand")
    r, path = sample_paths(m, d, idx, [[0.1], [0.01]])
    assert r["full_hand_collision_geoms"] == 2
    assert r["categories"]["table"]["penetration_m"] > 0.009
    assert r["categories"]["table"]["witness"]["hand_body"] == "hand"
    assert r["within_joint_limits"] and not r["continuous_collision_certified"]
    assert np.allclose(d.qpos, 0) and len(path) >= 46
