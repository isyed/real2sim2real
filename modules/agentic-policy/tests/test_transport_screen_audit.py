from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from real2sim2real.agentic_policy.transport_screen_audit import audit_transport_screen


@pytest.mark.parametrize("obstacle", [False, True])
def test_carried_object_clears_static_object_conflict_without_mutating_state(obstacle):
    wall = (
        '<geom name="wall" type="box" pos=".21 0 0" size=".05 .05 .05"/>'
        if obstacle
        else ""
    )
    m = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>"""
        + wall
        + """
      <body name="hand"><joint name="x" type="slide" axis="1 0 0"/><geom name="pad" type="box" size=".05 .05 .05" mass="1"/></body>
      <body name="object" pos=".1 0 0"><freejoint/><geom name="object_geom" type="box" size=".05 .05 .05" mass="1"/></body>
      </worldbody></mujoco>"""
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    saved = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
    s = SimpleNamespace(
        model=m,
        data=d,
        mj=mujoco,
        object=m.body("object").id,
        wrist=m.body("hand").id,
        robot={m.body("hand").id},
        qids=np.array([0]),
        nominal=np.array([0.0]),
        feedback=0.01,
        _vector=lambda q: np.array([q["x"]]),
    )
    result = audit_transport_screen(s, [{"x": 0.02}], [1.0])
    assert result["groups"]["stationary_object"]["robot_object"][
        "peak_penetration_m"
    ] == pytest.approx(0.02)
    assert (
        result["groups"]["rigidly_carried_object"]["robot_object"]["peak_penetration_m"]
        < 1e-12
    )
    current = saved.copy()
    mujoco.mj_getState(m, d, current, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(current, saved)
    assert not result["path_admitted"] and result["live_physics_steps"] == 0
    if obstacle:
        # A carried-object assumption must still expose world-obstacle contact.
        assert result["groups"]["rigidly_carried_object"]["object_environment"][
            "peak_penetration_m"
        ] == pytest.approx(0.01)
        assert (
            result["groups"]["stationary_object"]["object_environment"][
                "peak_penetration_m"
            ]
            == 0
        )
    with pytest.raises(ValueError, match="feedback ticks"):
        audit_transport_screen(s, [{"x": 0.02}], [0.001])
