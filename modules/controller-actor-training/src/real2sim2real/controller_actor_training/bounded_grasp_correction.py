"""Pinned, bounded robot-command probes on a continuous grasp/return protocol.

Corrections are declared before simulation and recover to zero before release.
They change neither object state nor reference motion. They are research probes,
not stationary-to-motion promotion certificates or learned policies.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from .paired_acquisition_audit import pinned_json
from real2sim2real.artifacts import digest
from .closing_wrist_offsets import validate as validate_wrist, fraction


def validate(spec, config):
    if set(spec) != {
        "baseline",
        "vectors_object_m",
        "rotation_vectors_object_rad",
        "finger_joint_offsets_rad",
    }:
        raise ValueError(
            "Pinned baseline and explicit bilateral wrist/finger corrections required"
        )
    validate_wrist(
        {k: spec[k] for k in ("vectors_object_m", "rotation_vectors_object_rad")}
    )
    offsets = spec["finger_joint_offsets_rad"]
    if not isinstance(offsets, dict) or any(
        not n.endswith(("MCP_FE", "PIP", "DIP", "thumb_IP"))
        or not np.isfinite(v)
        or abs(v) > 0.06
        for n, v in offsets.items()
    ):
        raise ValueError(
            "Finite named finger-flexion corrections bounded by 0.06 rad required"
        )
    baseline = pinned_json(spec["baseline"])
    source = Path(spec["baseline"]["path"])
    for name, field in (("states.npz", "states_sha256"), ("scene.xml", "scene_sha256")):
        if digest(source.with_name(name)) != baseline[field]:
            raise ValueError("Frozen correction source physical artifacts changed")
    if (
        not baseline.get("physics_run")
        or baseline.get("motion_condition") != "demonstrated"
        or baseline.get("residual_adapter")
        or baseline.get("bounded_grasp_correction")
        or "incomplete_pilot" in baseline["rejected_for"]
        or not baseline.get("continuous_protocol", {}).get("release_planned")
    ):
        raise ValueError(
            "Complete uncorrected continuous physics source with release required"
        )
    if config.get("motion_condition", "demonstrated") not in (
        "demonstrated",
        "stationary_hold",
    ):
        raise ValueError(
            "Choose demonstrated motion or a complete stationary diagnostic"
        )
    ignored = {"output", "purpose", "bounded_grasp_correction", "motion_condition"}
    strip = lambda c: {k: v for k, v in c.items() if k not in ignored}
    if strip(config) != strip(baseline["config"]):
        raise ValueError(
            "Correction probes cannot retune the frozen controller, clock or physics"
        )
    return baseline


def amplitude(time, live):
    return command_amplitude(time, live.rollout, live.correction_protocol)


def command_amplitude(time, rollout, protocol):
    p = protocol
    if p.get("release_disabled_for_stationary_diagnostic"):
        time = min(time, p["capture_duration_s"])
    return fraction(
        time, rollout, p["capture_duration_s"], p["finger_release"]["start_s"]
    )


def bind_fingers(live, model, spec, protocol):
    """Bind atomically to the existing owner; never write physical coordinates."""
    offsets = spec["finger_joint_offsets_rad"]
    by_name = {}
    for pad in live.servo.pads:
        for q, dof, limits in zip(pad["qids"], pad["dofs"], pad["limits"]):
            name = model.joint(int(model.dof_jntid[dof])).name
            by_name[name] = (int(q), np.asarray(limits, float))
    if set(offsets) - set(by_name):
        raise ValueError("Posture correction may own admitted finger joints only")
    changes = []
    for hand in live.hands:
        delta = np.array(
            [
                offsets.get(next(n for n, (qid, _) in by_name.items() if qid == q), 0)
                for q in hand["qids"]
            ],
            float,
        )
        if np.linalg.norm(delta) > 0.12 + 1e-12:
            raise ValueError("Per-hand posture correction norm exceeds 0.12 rad")
        shortest_ramp = min(
            live.rollout["close_s"],
            protocol["finger_release"]["start_s"] - protocol["capture_duration_s"],
        )
        if np.max(abs(delta), initial=0) * 1.875 / shortest_ramp > 0.2 + 1e-12:
            raise ValueError("Posture correction speed exceeds 0.2 rad/s")
        # Check both closure and every transition endpoint against native bounds.
        for q, value in zip(hand["qids"], delta):
            limits = next(lim for qid, lim in by_name.values() if qid == q)
            k = list(hand["qids"]).index(q)
            if any(
                not limits[0] - 1e-10 <= point[k] + value <= limits[1] + 1e-10
                for point in hand["waypoints"]
            ):
                raise ValueError(
                    "Posture correction exceeds native finger/control limits"
                )
        changes.append((hand, delta))
    for hand, delta in changes:
        hand["bounded_posture_offset_rad"] = delta
    live.correction_protocol = protocol


def finger_command(live, time, nominal):
    if not hasattr(live, "correction_protocol"):
        return nominal
    a = amplitude(time, live)
    if a == 0:
        return nominal
    for hand in live.hands:
        delta = hand["bounded_posture_offset_rad"]
        if np.any(delta):
            nominal[hand["qids"]] += a * delta
    return nominal


def wrist_targets(
    model,
    data,
    targets,
    times,
    qids,
    vids,
    arms,
    roots,
    parent,
    spec,
    speed_limits,
    margin,
):
    """Small damped IK in scratch state; admit complete bounded command paths."""
    import mujoco
    from .arm_object_feedback import joint_offset
    from .acquisition_profile import screen

    result = np.asarray(targets, float).copy()
    nonzero = any(
        np.any(spec[k][r])
        for k in ("vectors_object_m", "rotation_vectors_object_rad")
        for r in ("right", "left")
    )
    residual_position = residual_angle = largest_joint_change = 0.0
    if nonzero:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        p = parent["continuous_protocol"]
        ranges = model.jnt_range[
            [model.joint(n).id for arm in arms for n in arm["joints"]]
        ]
        for i, t in enumerate(times):
            a = command_amplitude(
                t,
                parent["configuration"]["rollout"],
                p,
            )
            if a == 0:
                continue
            obj_rotation = np.asarray(parent["records"][i]["reference_rotation"])
            scratch.qpos[qids] = targets[i]
            mujoco.mj_fwdPosition(model, scratch)
            goals = {}
            for arm in arms:
                name, body = arm["name"], roots[arm["name"]]
                goals[name] = (
                    scratch.xpos[body].copy()
                    + a * obj_rotation @ spec["vectors_object_m"][name],
                    Rotation.from_rotvec(
                        a * obj_rotation @ spec["rotation_vectors_object_rad"][name]
                    ).as_matrix()
                    @ scratch.xmat[body].reshape(3, 3),
                )
            start = 0
            for arm in arms:
                name, width = arm["name"], len(arm["joints"])
                sl = slice(start, start + width)
                body = roots[name]
                for _ in range(16):
                    mujoco.mj_fwdPosition(model, scratch)
                    error = np.r_[
                        goals[name][0] - scratch.xpos[body],
                        Rotation.from_matrix(
                            goals[name][1] @ scratch.xmat[body].reshape(3, 3).T
                        ).as_rotvec(),
                    ]
                    if (
                        np.linalg.norm(error[:3]) < 1e-5
                        and np.linalg.norm(error[3:]) < 1e-4
                    ):
                        break
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, scratch, jp, jr, body)
                    scratch.qpos[qids[sl]] += joint_offset(
                        np.vstack([jp, jr])[:, vids[sl]],
                        error,
                        damping=0.003,
                        max_offset_rad=0.04,
                    )
                    scratch.qpos[qids[sl]] = np.clip(
                        scratch.qpos[qids[sl]],
                        ranges[sl, 0] + margin,
                        ranges[sl, 1] - margin,
                    )
                mujoco.mj_fwdPosition(model, scratch)
                residual_position = max(
                    residual_position,
                    float(np.linalg.norm(goals[name][0] - scratch.xpos[body])),
                )
                residual_angle = max(
                    residual_angle,
                    float(
                        Rotation.from_matrix(
                            goals[name][1] @ scratch.xmat[body].reshape(3, 3).T
                        ).magnitude()
                    ),
                )
                start += width
            result[i] = scratch.qpos[qids]
            largest_joint_change = max(
                largest_joint_change, float(np.linalg.norm(result[i] - targets[i]))
            )
    admission = screen(model, qids, result, times, speed_limits, margin)
    protocol = parent["continuous_protocol"]
    stationary = protocol.get("release_disabled_for_stationary_diagnostic", False)
    release_index = max(
        0, int(np.searchsorted(times, protocol["finger_release"]["start_s"]) - 1)
    )
    release_targets_equal = (
        np.array_equal(result[release_index:], targets[release_index:])
        if not stationary
        else None
    )
    admission.update(
        maximum_ik_position_error_m=residual_position,
        maximum_ik_angle_error_rad=residual_angle,
        maximum_arm_joint_change_rad=largest_joint_change,
        physical_state_written=False,
        corrections_zero_before_release=release_targets_equal,
        stationary_corrections_retained=bool(stationary),
    )
    admission["accepted"] &= (
        residual_position <= 0.0003
        and residual_angle <= 0.003
        and largest_joint_change <= 0.15
        and release_targets_equal is not False
    )
    return result, admission
