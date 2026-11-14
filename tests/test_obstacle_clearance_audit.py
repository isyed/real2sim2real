import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from real2sim2real.controller_actor_training.obstacle_clearance_audit import (
    clearance_witness,
)


def scene():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="obstacle" type="box" size=".1 .1 .1"/>
    <body pos=".25 0 0"><freejoint/><geom name="moving" type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_signed_witness_pruning_and_no_state_mutation():
    m, d = scene()
    saved = d.qpos.copy(), d.qvel.copy(), d.time, m.geom_contype.copy()
    row = clearance_witness(m, d, ["moving"], ["obstacle"], search_m=0.1)
    assert row["complete_within_search"]
    assert row["nearest"]["signed_distance_m"] == pytest.approx(0.05)
    assert clearance_witness(m, d, ["moving"], ["obstacle"], search_m=0.02)[
        "no_pair_within_search"
    ]
    np.testing.assert_array_equal(d.qpos, saved[0])
    np.testing.assert_array_equal(d.qvel, saved[1])
    assert d.time == saved[2]
    np.testing.assert_array_equal(m.geom_contype, saved[3])
    d.qpos[0] = 0.19
    mujoco.mj_fwdPosition(m, d)
    assert clearance_witness(m, d, ["moving"], ["obstacle"])["nearest"][
        "signed_distance_m"
    ] == pytest.approx(-0.01)


def test_malformed_distance_witness_is_unknown_not_clear(monkeypatch):
    m, d = scene()

    def bad_distance(model, data, g, h, radius, segment):
        segment[:] = [0, 0, 0, 10, 0, 0]
        return 0.0

    monkeypatch.setattr(mujoco, "mj_geomDistance", bad_distance)
    row = clearance_witness(m, d, ["moving"], ["obstacle"], search_m=0.1)
    assert row["invalid_query_count"] == 1
    assert not row["complete_within_search"]
    assert not row["no_pair_within_search"]
    assert row["nearest"] is None
