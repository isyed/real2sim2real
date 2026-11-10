"""Kinematic hold audit of saved physics; original verdicts remain unchanged."""

import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .grasp_feedback import relative_translation_drift
from .object_grasp_pilot import (
    load_scene,
    set_wrist_frame,
    set_root_frame,
    object_transform,
)


def penetration_witnesses(model, data, robot_geoms):
    """Name actual intersecting geometry pairs; no inferred contact forces."""
    result = []
    for contact in data.contact:
        pair = sorted([int(contact.geom1), int(contact.geom2)])
        if contact.dist >= 0 or not set(pair) & set(robot_geoms):
            continue
        result.append(
            dict(
                geom_ids=pair,
                geoms=[model.geom(g).name or f"geom-{g}" for g in pair],
                bodies=[model.body(int(model.geom_bodyid[g])).name for g in pair],
                penetration_m=float(-contact.dist),
                position_world_m=np.asarray(contact.pos).tolist(),
            )
        )
    return result


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
    robot_geoms = set(idx["geoms"])
    if "support" in candidate:
        hand = candidate["support"]
        set_root_frame(
            model,
            data,
            hand["root_body"],
            hand["fixture_prefix"],
            hand["base_position_m"],
            hand["base_rotation"],
        )
        robot_geoms.update(robot_indices(model, hand["root_body"])["geoms"])
    states = dict(np.load(directory / "states.npz"))
    if len(states["qpos"]) != len(result["records"]):
        raise ValueError("State/record count differs")
    objects, wrists, rotations = [], [], []
    peaks = {}
    for q, time, row in zip(states["qpos"], states["timestamps_s"], result["records"]):
        if time != row["time_s"]:
            raise ValueError("State/record time differs")
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for witness in penetration_witnesses(model, data, robot_geoms):
            key = tuple(witness["geom_ids"])
            if witness["penetration_m"] > peaks.get(key, {}).get("penetration_m", 0):
                peaks[key] = dict(witness, time_s=float(time), phase=row["phase"])
        if row["phase"] != "hold":
            continue
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
        sampled_penetration_witnesses=sorted(
            peaks.values(), key=lambda w: w["penetration_m"], reverse=True
        ),
        sampled_penetration_is_continuous_bound=False,
        note="Coordinate and saved-sample geometry audit only; no new physical evidence or changed admission thresholds.",
    )
    write_json(output, report)
    return report
