from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.controller_actor_training.release_path_search import (
    contact_profile,
    matching_geom,
    bounded_rotation,
)


def test_path_contact_constraints_preserve_intended_secondary_object_contact():
    model = SimpleNamespace(geom=lambda g: SimpleNamespace(name=str(g)))
    data = SimpleNamespace(
        contact=[
            SimpleNamespace(geom1=1, geom2=3, dist=-0.001),
            SimpleNamespace(geom1=1, geom2=2, dist=-0.004),
            SimpleNamespace(geom1=2, geom2=3, dist=-0.01),
        ]
    )
    objects, robot, gap, witness = contact_profile(model, data, {1}, {1, 2}, {3}, 0.002)
    np.testing.assert_allclose(objects, [0.003])
    np.testing.assert_allclose(robot, [0.001, 0.001])
    assert gap == -0.001
    assert witness["geoms"] == ["1", "2"] and not witness["object_contact"]


def test_unnamed_colliders_match_by_body_and_compiled_shape_without_visual_confusion():
    m = SimpleNamespace(
        geom=lambda g: SimpleNamespace(name=""),
        body=lambda b: SimpleNamespace(name="hand"),
        geom_bodyid=[1],
        geom_type=[7],
        geom_contype=[1],
        geom_conaffinity=[1],
        geom_size=np.ones((1, 3)),
        geom_pos=np.zeros((1, 3)),
    )
    original = SimpleNamespace(
        body=lambda b: SimpleNamespace(id=1),
        ngeom=2,
        geom_bodyid=[1, 1],
        geom_type=[7, 7],
        geom_contype=[0, 1],
        geom_conaffinity=[0, 1],
        geom_size=np.ones((2, 3)),
        geom_pos=np.zeros((2, 3)),
    )
    assert matching_geom(m, original, 0) == 1
    original.geom_contype[0] = original.geom_conaffinity[0] = 1
    with pytest.raises(ValueError, match="unambiguous"):
        matching_geom(m, original, 0)


def test_position_only_margin_query_detects_near_contact_without_advancing_physics():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="hand" pos=".023 0 0"><joint type="slide"/><geom name="pad" size=".01"/></body>
    <body name="object"><freejoint/><geom name="object_geom" size=".01" mass=".5"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_fwdPosition(model, data)
    assert not data.ncon
    before = (data.qpos.copy(), data.qvel.copy(), data.time)
    pad = model.geom("pad").id
    obj = model.geom("object_geom").id
    model.geom_margin[pad] = 0.005
    mj.mj_fwdPosition(model, data)
    error, _, gap, _ = contact_profile(model, data, {pad}, {pad}, {obj}, 0.005)
    assert gap == pytest.approx(0.003)
    np.testing.assert_allclose(error, [0.002])
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]


def test_rotation_optimization_parameters_cannot_escape_the_angular_budget():
    for values in [[0.5, 0, 0], [1, 1, 1], [-1, 1, 0]]:
        angle = bounded_rotation(values)
        assert np.linalg.norm(angle) < 0.2
        assert np.dot(angle, values) > 0
    np.testing.assert_allclose(bounded_rotation([0.5, 0, 0]), [0.1, 0, 0], atol=1e-9)
    with pytest.raises(ValueError):
        bounded_rotation([float("nan"), 0, 0])
