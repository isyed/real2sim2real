"""Kinematic hold audit of saved physics; original verdicts remain unchanged."""

import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .grasp_feedback import relative_translation_drift
from .object_grasp_pilot import load_scene, set_wrist_frame, object_transform


def audit(config, directory, output):
    import mujoco
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError("Preserve existing hold audits")
    result = json.loads((directory / "result.json").read_text())
    if digest(directory / "states.npz") != result["states_sha256"]:
        raise ValueError("Saved physical states changed")
    if (
        digest(Path(config["output"]) / "scene/scene.xml")
        != result["source_scene_sha256"]
    ):
        raise ValueError("Saved states belong to another scene")
    model, data, _ = load_scene(config)
    candidate = result["candidate"]
    set_wrist_frame(
        model, data, config, candidate["base_position_m"], candidate["base_rotation"]
    )
    idx = robot_indices(model, config["robot"]["root_body"])
    states = dict(np.load(directory / "states.npz"))
    if len(states["qpos"]) != len(result["records"]):
        raise ValueError("State/record count differs")
    objects, wrists, rotations = [], [], []
    for q, time, row in zip(states["qpos"], states["timestamps_s"], result["records"]):
        if time != row["time_s"]:
            raise ValueError("State/record time differs")
        if row["phase"] != "hold":
            continue
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj, wrist = object_transform(model, data)[:3, 3], data.xpos[idx["root"]]
        if not np.allclose(
            obj, row["object_position_m"], atol=1e-10, rtol=0
        ) or not np.allclose(wrist, row["wrist_position_m"], atol=1e-10, rtol=0):
            raise ValueError("Recorded object/wrist position does not reproduce")
        objects.append(obj.copy())
        wrists.append(wrist.copy())
        rotations.append(data.xmat[idx["root"]].reshape(3, 3).copy())
    corrected = relative_translation_drift(objects, wrists, rotations)
    failures = [r for r in result["rejected_for"] if r != "hold_slip"]
    if corrected > 0.01:
        failures.append("hold_slip")
    report = dict(
        schema="real2sim2real.grasp-hold-audit/1",
        source_result_sha256=digest(directory / "result.json"),
        source_states_sha256=result["states_sha256"],
        original_success=result["success"],
        original_hold_slip_m=result["hold_slip_m"],
        corrected_wrist_frame_drift_m=corrected,
        recomputed_rejected_for=failures,
        would_pass_corrected_metric=not failures,
        original_results_changed=False,
        physics_rerun=False,
        independent_validation=False,
        note="Coordinate correction only; this does not provide new physical evidence.",
    )
    write_json(output, report)
    return report
