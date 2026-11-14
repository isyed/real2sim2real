import numpy as np
import pytest
from real2sim2real.controller_actor_training.contact_patch_motion import (
    ContactPatchMotion,
)
from real2sim2real.controller_actor_training.mujoco_step_trace import MujocoStepTrace
from real2sim2real.controller_actor_training.traced_contact_motion import audit_trace


def test_trace_audit_reproduces_forces_and_does_not_step_or_mutate_live_data(
    tmp_path, monkeypatch
):
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string("""<mujoco><option timestep=".001" integrator="implicitfast"/>
    <worldbody><geom name="pad" type="plane" size="1 1 .1"/>
    <body name="object" pos="0 0 .049"><freejoint/><geom name="ball" size=".05" mass=".2"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    d.qvel[0] = 0.02
    with MujocoStepTrace(m, d, tmp_path / "trace", window_s=0.1):
        for _ in range(30):
            mj.mj_step(m, d)
    gid = m.geom("pad").id
    settings = {
        "minimum_force_n": 0.01,
        "sliding_speed_m_s": 0.001,
        "rolling_speed_rad_s": 0.02,
        "maximum_sample_gap_s": 0.05,
    }
    audit = ContactPatchMotion(
        m,
        m.body("object").id,
        {"hand": [gid]},
        [{"gid": gid, "role": "hand", "group": "pad"}],
        settings,
    )
    before = d.qpos.copy(), d.qvel.copy(), d.time
    monkeypatch.setattr(
        mj, "mj_step", lambda *a: pytest.fail("Audit must not step physics")
    )
    result = audit_trace(m, tmp_path / "trace", audit, stride=1)
    assert result["force_reconstruction_max_error_n"] < 1e-10
    assert result["summary"]["pads"]["hand:pad"]["sampled_sliding_path_m"] > 0
    assert not result["acceptance_claim"]
    strided = audit_trace(m, tmp_path / "trace", audit, stride=7)
    assert len(strided["records"]) == 6
    assert strided["records"][-1]["time_s"] == result["records"][-1]["time_s"]
    np.testing.assert_array_equal(d.qpos, before[0])
    np.testing.assert_array_equal(d.qvel, before[1])
    assert d.time == before[2]
    # Re-solving under changed physics must expose a force mismatch rather than
    # silently certify the trace as reproducing its original contact response.
    m.opt.gravity[2] *= 1.2
    changed = audit_trace(m, tmp_path / "trace", audit, stride=1)
    assert changed["force_reconstruction_max_error_n"] > .01
    with pytest.raises(ValueError, match="stride"):
        audit_trace(m, tmp_path / "trace", audit, stride=0)
