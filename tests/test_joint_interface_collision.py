import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco")

from real2sim2real.controller_actor_training.joint_interface_collision import (
    exclude_joint_interface,
)


def test_explicit_interface_exclusion_preserves_other_contacts():
    root = ET.fromstring("""<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size=".1" pos=".15 0 0"/>
      <body name="base"><geom type="sphere" size=".1"/>
        <body name="shoulder"><joint axis="0 0 1"/>
          <geom type="sphere" size=".1" pos=".1 0 0"/>
          <body name="elbow" pos=".3 0 0"><joint axis="0 1 0"/>
            <geom type="sphere" size=".05"/>
          </body>
        </body>
      </body></worldbody></mujoco>""")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def pairs(m):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return {
            frozenset(m.body(int(m.geom_bodyid[g])).name for g in [c.geom1, c.geom2])
            for c in d.contact
        }

    assert frozenset(["base", "shoulder"]) in pairs(model)
    with pytest.raises(ValueError):
        exclude_joint_interface(root, model, "base", "elbow")
    with pytest.raises(ValueError):
        exclude_joint_interface(root, model, "world", "shoulder")
    assert exclude_joint_interface(root, model, "base", "shoulder")
    assert not exclude_joint_interface(root, model, "shoulder", "base")
    corrected = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert frozenset(["base", "shoulder"]) not in pairs(corrected)
    assert frozenset(["world", "shoulder"]) in pairs(corrected)
