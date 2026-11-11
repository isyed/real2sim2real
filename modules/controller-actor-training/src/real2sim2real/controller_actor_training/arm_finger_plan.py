"""Bind independently admitted finger plans while preserving audited wrist paths."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json


def validate_hand_change(original, proposed, old_poses, new_poses):
    for key in ("joint_names", "root_body", "fixture_prefix"):
        if original[key] != proposed[key]:
            raise ValueError("Finger replanning must preserve hand identity")
    names = original["joint_names"]
    fixture = np.array([n.startswith(original["fixture_prefix"]) for n in names])
    for key in ("q_pregrasp", "q_open", "q_grasp", "q_closed"):
        before, after = (
            np.asarray(original[key], float),
            np.asarray(proposed[key], float),
        )
        if (
            before.shape != (len(names),)
            or after.shape != before.shape
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
            or np.max(abs(after[~fixture] - before[~fixture]), initial=0) > 0.35 + 1e-6
        ):
            raise ValueError("Only bounded owned finger posture changes are allowed")
        if key in ("q_pregrasp", "q_open") and not np.allclose(
            before[~fixture], after[~fixture], atol=1e-6, rtol=0
        ):
            raise ValueError("Retain the original finger acquisition waypoints")
        p, r = old_poses[key]
        q, s = new_poses[key]
        if any(not np.isfinite(x).all() for x in (p, r, q, s)):
            raise ValueError("Finite wrist poses required")
        if np.linalg.norm(p - q) > 1e-6 or np.linalg.norm(r - s) > 1e-6:
            raise ValueError("Replanned fingers cannot change audited wrist waypoints")


def wrist_poses(config, hand):
    import mujoco
    from .object_grasp_pilot import load_scene, set_root_frame
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    model, data, _ = load_scene(config)
    set_root_frame(
        model,
        data,
        hand["root_body"],
        hand["fixture_prefix"],
        hand["base_position_m"],
        hand["base_rotation"],
    )
    indices = robot_indices(model, hand["root_body"])
    result = {}
    for key in ("q_pregrasp", "q_open", "q_grasp", "q_closed"):
        data.qpos[indices["q"]] = hand[key]
        mujoco.mj_fwdPosition(model, data)
        result[key] = (
            data.xpos[indices["root"]].copy(),
            data.xmat[indices["root"]].reshape(3, 3).copy(),
        )
    return result


def bind(parent, spec, output):
    from .object_grasp_pilot import load_plan

    path = Path(spec["plan"])
    if digest(path) != spec["sha256"]:
        raise ValueError("Pinned replanned fingers changed")
    plan = json.loads(path.read_text())
    if plan.get("source_plan_sha256") != parent["candidate"]["bilateral_plan_sha256"]:
        raise ValueError("Finger proposal does not derive from this audited grasp")
    rows = [c for c in plan["candidates"] if c["candidate_id"] == spec["candidate_id"]]
    if (
        len(rows) != 1
        or rows[0]["accepted"] is not True
        or rows[0]["gravity_wrench"]["feasible"] is not True
        or rows[0].get("wrists_frozen") is not True
        or not all(p["all_pads_observed"] for p in rows[0]["patches"])
    ):
        raise ValueError("An admitted fixed-wrist finite-patch finger plan is required")
    base = parent["configuration"]
    original_plan = load_plan(base)
    if plan["settling"] != original_plan["settling"]:
        raise ValueError("Finger planning must preserve task initialization")
    original = [deepcopy(parent["candidate"]), deepcopy(parent["candidate"]["support"])]
    original[0]["root_body"] = base["robot"]["root_body"]
    original[0]["fixture_prefix"] = "fixture_"
    hands = deepcopy(rows[0]["hands"])
    if len(hands) != 2 or len(rows[0]["patches"]) != 2:
        raise ValueError("Two complete admitted hands required")
    for a, b in zip(original, hands):
        validate_hand_change(a, b, wrist_poses(base, a), wrist_poses(base, b))
    candidate = deepcopy(parent["candidate"])
    for key in (
        "base_position_m",
        "base_rotation",
        "joint_names",
        "q_pregrasp",
        "q_open",
        "q_grasp",
        "q_closed",
        "targets_m",
        "site_errors_m",
        "pad_plan",
    ):
        candidate[key] = deepcopy(hands[0][key])
    candidate.update(
        candidate_id=spec["candidate_id"],
        support=hands[1],
        bilateral_plan_sha256=digest(path),
        accepted=True,
        rejected_for=[],
    )
    out = Path(output) / "finger-planning"
    out.mkdir()
    write_json(
        out / "plan.json", dict(original_plan, candidates=[candidate], accepted=1)
    )
    result = deepcopy(parent)
    result["candidate"] = candidate
    result["configuration"]["planning_directory"] = str(out)
    return result, dict(
        plan=str(path),
        sha256=digest(path),
        candidate_id=spec["candidate_id"],
        audited_wrists_preserved=True,
        physical_retention_validated=False,
    )


def validate_stationary(config):
    diagnostic = bool(config.get("diagnostic_stationary"))
    if diagnostic and config.get("validated_stationary"):
        raise ValueError("Choose diagnostic replay or stationary promotion, not both")
    spec = (
        config.get("diagnostic_stationary")
        if diagnostic
        else config.get("validated_stationary")
    )
    if not spec:
        raise ValueError(
            "New finger-plan motion requires a complete passing stationary arm hold"
        )
    path = Path(spec["result"])
    if digest(path) != spec["sha256"]:
        raise ValueError("Pinned stationary arm hold changed")
    report = json.loads(path.read_text())
    if report.get("residual_adapter") is not None:
        raise ValueError(
            "Residual stationary holds require policy-aware promotion; do not silently omit the actor"
        )
    if (
        report.get("schema") != "real2sim2real.arm-replay-pilot/1"
        or (not diagnostic and report["success"] is not True)
        or (
            diagnostic
            and (
                not report.get("physics_run")
                or "incomplete_pilot" in report.get("rejected_for", [])
            )
        )
        or report.get("motion_condition") != "stationary_hold"
        or digest(path.with_name("states.npz")) != report["states_sha256"]
        or digest(path.with_name("scene.xml")) != report["scene_sha256"]
    ):
        raise ValueError("Passing unchanged stationary physics artifacts required")
    allowed = {
        "output",
        "deadline_s",
        "purpose",
        "motion_condition",
        "validated_stationary",
        "diagnostic_stationary",
    }
    strip = lambda x: {k: v for k, v in x.items() if k not in allowed}
    if strip(config) != strip(report["config"]):
        raise ValueError(
            "Stationary-to-motion promotion cannot retune clock, posture, controller or physics"
        )
    audit_path = Path(config["audit"])
    if digest(audit_path) != config["audit_sha256"]:
        raise ValueError("Pinned arm path changed")
    audit = json.loads(audit_path.read_text())
    source = Path(audit["config"]["trial"])
    if digest(source) != audit["sources"][str(source)]:
        raise ValueError("Source command duration changed")
    expected = json.loads(source.read_text())["records"]
    # The shared rollout ends at the nearest integration-grid point to the
    # declared endpoint. Native capture clocks need not divide the physics step.
    timestep = report.get("physics_timestep_s", config.get("timestep_s", 0.000001))
    if not np.isfinite(timestep) or not 0 < timestep <= 0.01:
        raise ValueError("Finite recorded physics clock required for promotion")
    endpoint_tolerance = max(1e-6, 0.5 * timestep + 1e-9)
    if (
        len(report["records"]) != len(expected)
        or not np.isfinite(report["records"][-1]["time_s"])
        or abs(report["records"][-1]["time_s"] - expected[-1]["time_s"])
        > endpoint_tolerance
    ):
        raise ValueError(
            "Stationary promotion must preserve the complete declared command clock"
        )
    return dict(
        result=str(path),
        sha256=digest(path),
        **(
            dict(
                diagnostic_only=True,
                promotion_eligible=False,
                stationary_success=report["success"],
                stationary_rejected_for=report["rejected_for"],
            )
            if diagnostic
            else {}
        ),
    )


def path_screen(
    model,
    data,
    targets,
    times,
    qids,
    live,
    arm_geoms,
    robot_geoms,
    object_geoms,
    criteria,
):
    """New finger commands against the full arm/environment path in scratch data."""
    import mujoco
    from .arm_path_audit import joint_neighbors
    from .arm_finger_feedback import finger_waypoint

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    maximum_arm = maximum_robot = 0.0
    worst_contact = None
    release_clearances = []
    for time, target in zip(times, targets):
        scratch.qpos[qids] = target
        for hand in live.hands:
            scratch.qpos[hand["qids"]] = finger_waypoint(
                time,
                *hand["waypoints"],
                live.rollout,
                release_open=hand.get("release_open"),
                release_fast_mask=hand.get("release_fast_mask"),
                closing_windows=hand.get("closing_windows"),
            )
        if hasattr(live, "correction_protocol"):
            from .bounded_grasp_correction import finger_command

            finger_command(live, time, scratch.qpos)
        mujoco.mj_fwdPosition(model, scratch)
        if (
            live.rollout.get("finger_release")
            and time >= live.rollout["finger_release"]["start_s"]
        ):
            release_clearances.append(
                hand_table_clearance(
                    model, scratch, robot_geoms - arm_geoms, model.geom("table").id
                )
            )
        for contact in scratch.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & object_geoms or not pair & robot_geoms:
                continue
            bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if pair & arm_geoms and joint_neighbors(model, *bodies):
                continue
            if -float(contact.dist) > maximum_robot:
                maximum_robot = -float(contact.dist)
                worst_contact = dict(
                    time_s=float(time),
                    geoms=[model.geom(g).name for g in (contact.geom1, contact.geom2)],
                    penetration_m=maximum_robot,
                )
            if pair & arm_geoms:
                maximum_arm = max(maximum_arm, -float(contact.dist))
    accepted = (
        maximum_arm <= criteria["arm_penetration_m"]
        and maximum_robot <= criteria["robot_penetration_m"]
    )
    report = dict(
        accepted=accepted,
        samples=len(times),
        maximum_arm_penetration_m=maximum_arm,
        maximum_robot_penetration_m=maximum_robot,
        object_contacts_excluded=True,
        pressure_offsets_not_predicted=True,
        physical_retention_validated=False,
        worst_robot_contact=worst_contact,
        minimum_release_hand_table_clearance_m=min(release_clearances)
        if release_clearances
        else None,
        table_clearance_method="conservative collision-geometry height above the infinite tabletop upper envelope",
    )
    return report


def hand_table_clearance(model, data, hand_geoms, table):
    """Positive envelope gap guarantees table clearance; negative is conservative."""
    from .arm_clearance_audit import table_gap

    collidable = [
        g for g in hand_geoms if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    if not collidable:
        raise ValueError("Collidable hand geometries required for table clearance")
    return table_gap(model, data, collidable, table)
