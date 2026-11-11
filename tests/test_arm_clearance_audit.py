import pytest
from real2sim2real.controller_actor_training.arm_clearance_audit import table_contacts


def test_table_witnesses_exclude_object_support_and_identify_hand():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/>
    <body pos="0 0 .09"><freejoint/><geom name="hand" size=".1"/></body>
    <body pos=".3 0 .09"><freejoint/><geom name="object" size=".1"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_fwdPosition(model, data)
    rows = table_contacts(
        model, data, model.geom("table").id, {model.geom("hand").id: "left_hand"}
    )
    assert len(rows) == 1 and rows[0]["geometry"] == "hand"
    assert rows[0]["penetration_m"] == pytest.approx(0.01)
    assert rows[0]["role"] == "left_hand"
