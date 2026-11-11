import json

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.arm_residual_validate import compare


def reports(tmp_path):
    fields = (
        "controller_updates",
        "controller_period_s",
        "physics_timestep_s",
        "max_robot_penetration_m",
        "max_arm_penetration_m",
        "maximum_dynamic_velocity_ratio",
        "minimum_dynamic_joint_margin_rad",
    )
    paths = []
    for name in ("baseline", "candidate"):
        out = tmp_path / name
        out.mkdir()
        (out / "scene.xml").write_text("<mujoco/>")
        np.savez(
            out / "states.npz",
            qpos=np.array([[0.0, 1.0]]),
            qvel=np.array([[0.0, 0.0]]),
            ctrl=np.array([[1.0, 0.0]]),
            timestamps_s=[0.02],
        )
        report = dict.fromkeys(fields, 0)
        report.update(
            success=False,
            rejected_for=["hold_slip"],
            config=dict(output=str(out), gain=1),
            records=[dict(time_s=0.02)],
            solver_warning_counts={},
            torque_peak_nm={},
            live_finger_feedback_final={},
            states_sha256=digest(out / "states.npz"),
            scene_sha256=digest(out / "scene.xml"),
            hold_slip_m=0.011,
        )
        if name == "candidate":
            np.savez(
                out / "policy-trace.npz",
                observations=np.zeros((2, 2)),
                actions=np.zeros((1, 4)),
            )
            report["residual_adapter"] = dict(
                all_actions_zero=True, trace_sha256=digest(out / "policy-trace.npz")
            )
        path = out / "result.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_zero_verification_preserves_failed_task_and_rejects_one_bit_change(tmp_path):
    baseline, candidate = reports(tmp_path)
    result = compare(baseline, candidate)
    assert result["zero_reproduction_passed"] and not result["candidate_success"]
    path = candidate.with_name("states.npz")
    with np.load(path) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    arrays["qpos"][0, 1] = np.nextafter(1.0, 2.0)
    np.savez(path, **arrays)
    report = json.loads(candidate.read_text())
    report["states_sha256"] = digest(path)
    candidate.write_text(json.dumps(report))
    result = compare(baseline, candidate)
    assert not result["zero_reproduction_passed"]
    assert not result["state_arrays"]["qpos"]["equal"]


def test_changed_controller_or_nonzero_actions_cannot_claim_zero_replay(tmp_path):
    baseline, candidate = reports(tmp_path)
    report = json.loads(candidate.read_text())
    report["config"]["gain"] = 2
    candidate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration"):
        compare(baseline, candidate)
    report["config"]["gain"] = 1
    np.savez(
        candidate.with_name("policy-trace.npz"),
        observations=np.zeros((2, 2)),
        actions=np.ones((1, 4)),
    )
    report["residual_adapter"]["trace_sha256"] = digest(
        candidate.with_name("policy-trace.npz")
    )
    candidate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="zero-action"):
        compare(baseline, candidate)
