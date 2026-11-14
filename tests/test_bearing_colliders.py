import xml.etree.ElementTree as ET

import numpy as np
import pytest
from real2sim2real.retargeting.bearing_colliders import (
    append_bearing_colliders,
    triangle_prisms,
)


def test_patches_preserve_top_and_thickness_without_filling_global_hull():
    v = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [1, 1, 0], [1.1, 1, 0], [1, 1.1, 0]]
    )
    f = np.array([[0, 1, 2], [3, 4, 5]])
    p = triangle_prisms(v, f, thickness_m=0.004)
    np.testing.assert_allclose(p[:, :3], v[f])
    np.testing.assert_allclose(p[:, 3:, 2], -0.004)
    tree = ET.ElementTree(
        ET.fromstring(
            '<mujoco><worldbody><geom name="obstacle" type="box" size="1 1 1"/></worldbody></mujoco>'
        )
    )
    append_bearing_colliders(tree, v, f)
    assert len(tree.getroot().findall("asset/mesh")) == 2
    assert tree.getroot().find("worldbody/geom[@name='obstacle']") is not None
    with pytest.raises(ValueError):
        append_bearing_colliders(tree, v, f)


def test_degenerate_triangle_rejected_before_xml_mutation():
    tree = ET.ElementTree(ET.fromstring("<mujoco><worldbody/></mujoco>"))
    before = ET.tostring(tree.getroot())
    with pytest.raises(ValueError):
        append_bearing_colliders(tree, np.zeros((3, 3)), np.array([[0, 1, 2]]))
    assert ET.tostring(tree.getroot()) == before
