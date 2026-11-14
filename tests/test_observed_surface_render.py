import numpy as np
import pytest
from real2sim2real.retargeting.collision_visualization import CollisionRenderer


def test_observed_triangles_do_not_fill_holes_or_change_physics():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="coarse" type="box" size=".1 .1 .1"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry = model.geom_size.copy()
    triangles = np.array([[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]])
    renderer = CollisionRenderer(
        model,
        [0, 0, 0],
        hidden_geoms=["coarse"],
        static_surfaces=[(triangles, [220, 220, 220])],
    )
    assert not renderer.pieces
    np.testing.assert_array_equal(renderer.static_surfaces[0][0], triangles)
    assert renderer.frame(data).size == (900, 640)
    np.testing.assert_array_equal(model.geom_size, geometry)
    assert model.geom_contype[0] == 1
    with pytest.raises(ValueError):
        CollisionRenderer(model, [0, 0, 0], hidden_geoms=["missing"])
