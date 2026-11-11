import numpy as np
import pytest

from real2sim2real.controller_actor_training.finger_motor_audit import motor_state
from real2sim2real.controller_actor_training.finger_motor_audit import run


def test_effort_accounts_for_damping_and_both_motor_limits():
    q, v, c = np.array([0.0, 0.0]), np.array([1.0, -1.0]), np.array([0.3, -0.3])
    before = [a.copy() for a in (q, v, c)]
    r = motor_state(
        c,
        q,
        v,
        [10, 10],
        [[0, -10, -1], [0, -10, -1]],
        [[-0.5, 0.5], [-0.5, 0.5]],
        [[-0.2, 0.2], [-0.2, 0.2]],
    )
    np.testing.assert_allclose(r["effort_nm"], [0.5, -0.5])
    np.testing.assert_allclose(r["tracking_error_rad"], [0.2, -0.2])
    assert r["effort_clipped"].all() and r["command_clipped"].all()
    np.testing.assert_array_equal(r["effort_limit_fraction"], [1, 1])
    for old, now in zip(before, (q, v, c)):
        np.testing.assert_array_equal(old, now)


def test_tracking_error_does_not_imply_torque_saturation():
    r = motor_state([0.1], [0], [0], [0.9], [[0, -0.9, -0.04]], [[-2, 2]], [[-1, 1]])
    assert r["tracking_error_rad"][0] == 0.1
    assert r["effort_limit_fraction"][0] == pytest.approx(0.045)
    assert not r["effort_clipped"].any()
    with pytest.raises(ValueError):
        motor_state([np.nan], [0], [0], [0.9], [[0, -0.9, -0.04]], [[-2, 2]], [[-1, 1]])


def test_audit_effort_matches_mujoco_position_motor():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string("""<mujoco><worldbody><body><joint name="j"/>
        <geom size=".01"/></body></worldbody><actuator><position joint="j" kp="5"
        kv=".4" ctrlrange="-.2 .2" forcerange="-.5 .5"/></actuator></mujoco>""")
    d = mj.MjData(m)
    for q, v, c in [(0.01, 0.2, 0.04), (0, 0, 1), (0, 0, -1)]:
        d.qpos[0], d.qvel[0], d.ctrl[0] = q, v, c
        mj.mj_forward(m, d)
        r = motor_state(
            [c],
            [q],
            [v],
            m.actuator_gainprm[:, 0],
            m.actuator_biasprm[:, :3],
            m.actuator_forcerange,
            m.actuator_ctrlrange,
        )
        np.testing.assert_allclose(r["effort_nm"], d.actuator_force)


def test_saved_audit_pins_artifacts_and_never_changes_original_verdict(tmp_path):
    import json
    from real2sim2real.artifacts import digest

    mj = pytest.importorskip("mujoco")
    xml = """<mujoco><worldbody><body><joint name="finger"/>
        <geom size=".01"/></body></worldbody><actuator><position joint="finger"
        kp="5" kv=".4" ctrlrange="-.2 .2" forcerange="-.5 .5"/></actuator></mujoco>"""
    mj.MjModel.from_xml_string(xml)
    (tmp_path / "scene.xml").write_text(xml)
    np.savez(
        tmp_path / "states.npz",
        timestamps_s=[0, 0.02],
        qpos=[[0], [0.01]],
        qvel=[[0], [0.2]],
        ctrl=[[0.1], [0.1]],
    )
    source = dict(
        schema="real2sim2real.arm-replay-pilot/1",
        scene_sha256=digest(tmp_path / "scene.xml"),
        states_sha256=digest(tmp_path / "states.npz"),
        success=False,
        rejected_for=["hold_slip"],
        records=[dict(time_s=0, phase="close"), dict(time_s=0.02, phase="lift")],
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(source))
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    audit = run(path, tmp_path / "motor-audit")
    assert audit["source_success"] is False
    assert audit["source_rejected_for"] == ["hold_slip"]
    assert audit["acceptance_changed"] is False
    assert audit["joint_names"] == ["finger"]
    for name, content in original.items():
        assert (tmp_path / name).read_bytes() == content
    with pytest.raises(FileExistsError):
        run(path, tmp_path / "motor-audit")
    (tmp_path / "states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        run(path, tmp_path / "tampered")
