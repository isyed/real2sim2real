import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from real2sim2real.controller_actor_training.mujoco_step_trace import (
    MujocoStepTrace,
    audit_contact_geometry,
)


def model():
    return mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".001" integrator="implicitfast"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .05"><freejoint/><geom type="sphere" size=".05" mass=".2"/></body>
      <body pos=".3 0 .5"><joint name="hinge"/><geom type="capsule" size=".02 .1" mass=".1"/></body>
      </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")


def test_tracing_preserves_every_integration_state_and_keeps_bounded_tail(tmp_path):
    m = model()
    reference, traced = mujoco.MjData(m), mujoco.MjData(m)
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(m, signature))
    expected = []
    for i in range(80):
        reference.ctrl[:] = np.sin(i / 10) * 0.01
        mujoco.mj_step(m, reference)
        mujoco.mj_getState(m, reference, values, signature)
        expected.append(values.copy())
    original = mujoco.mj_step
    with MujocoStepTrace(m, traced, tmp_path / "trace", window_s=0.01) as trace:
        for i in range(80):
            traced.ctrl[:] = np.sin(i / 10) * 0.01
            mujoco.mj_step(m, traced)
            mujoco.mj_getState(m, traced, values, signature)
            np.testing.assert_array_equal(values, expected[i])
    assert mujoco.mj_step is original
    assert trace.fault is None
    with np.load(tmp_path / "trace/steps.npz") as a:
        assert len(a["indices"]) == 10
        assert a["indices"].tolist() == list(range(71, 81))
    assert (
        json.loads((tmp_path / "trace/summary.json").read_text())[
            "live_forward_calls_added"
        ]
        == 0
    )


def test_clock_reset_preserves_failing_input_and_restores_wrapper(
    tmp_path, monkeypatch
):
    m, calls = model(), []
    d = mujoco.MjData(m)
    original = mujoco.mj_step

    def reset_third(m, d):
        calls.append(1)
        if len(calls) == 3:
            mujoco.mj_resetData(m, d)
        else:
            original(m, d)

    monkeypatch.setattr(mujoco, "mj_step", reset_third)
    with MujocoStepTrace(m, d, tmp_path / "trace") as trace:
        for _ in range(3):
            mujoco.mj_step(m, d)
    assert trace.fault["before_s"] == 0.002 and trace.fault["after_s"] == 0
    assert not trace.fault["clock_ok"]
    assert (tmp_path / "trace/failure-forward.npz").is_file()
    assert mujoco.mj_step is reset_third


def test_contact_audit_detects_far_point_without_treating_plane_as_bounded():
    m = model()
    d = mujoco.MjData(m)
    d.qpos[0] = 100.0  # valid contact far from the plane's origin
    d.qpos[2] -= 0.001
    mujoco.mj_forward(m, d)
    assert d.ncon == 1
    assert not audit_contact_geometry(m, d)["anomalies"]
    d.contact[0].pos[0] += 1e7  # simulate a malformed collider output
    before = d.contact[0].pos.copy()
    report = audit_contact_geometry(m, d)
    assert len(report["anomalies"]) == 1
    assert report["anomalies"][0]["bounds"][0]["excess_m"] > 1e6
    np.testing.assert_array_equal(before, d.contact[0].pos)
    assert not report["contacts_modified"]
