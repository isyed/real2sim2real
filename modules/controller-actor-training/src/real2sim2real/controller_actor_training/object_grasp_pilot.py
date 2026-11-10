"""Finite CPU object-centric grasp planning and unassisted lift evaluation."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from real2sim2real.artifacts import digest, write_json
from .object_grasp_scene import quaternion


def load_scene(config):
    import mujoco

    root = Path(config["output"]) / "scene"
    receipt = json.loads((root / "receipt.json").read_text())
    task_source = receipt["config"]["task"]
    if digest(Path(config["task"])) != receipt["sources"][task_source]:
        raise ValueError("Task changed after scene preparation")
    if config["robot"]["root_body"] != receipt["config"]["robot"]["root_body"]:
        raise ValueError("Robot root differs from the prepared scene")
    if config["physics"] != receipt["config"]["physics"]:
        raise ValueError(
            "Nominal physics differs from the prepared scene; use explicit trial variants"
        )
    for name in (
        "finger_torque_limit_nm",
        "wrist_force_limit_n",
        "wrist_torque_limit_nm",
    ):
        if config["robot"][name] != receipt["config"]["robot"][name]:
            raise ValueError("Actuator limits differ from the prepared scene")
    if digest(root / "scene.xml") != receipt["scene_sha256"]:
        raise ValueError("Scene changed after preparation")
    for folder, records in [
        (root, receipt["assets"]),
        (root / "robot_assets", receipt["robot_assets"]),
    ]:
        for name, sha in records.items():
            if digest(folder / name) != sha:
                raise ValueError("Scene assets changed")
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    verify_unassisted_model(model)
    return (
        model,
        mujoco.MjData(model),
        trimesh.load(root / "object.obj", force="mesh", process=False),
    )


def verify_unassisted_model(model):
    import mujoco

    if model.neq or model.nmocap or not np.allclose(model.opt.gravity, [0, 0, -9.81]):
        raise ValueError(
            "Grasp pilot prohibits equality/mocap assistance or gravity changes"
        )
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task_object")
    if obj < 1:
        raise ValueError("Named dynamic task object required")
    joint = model.body_jntadr[obj]
    if (
        model.body_jntnum[obj] != 1
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        or model.body_mass[obj] <= 0
        or model.body_gravcomp[obj] != 0
        or model.jnt_stiffness[joint] != 0
    ):
        raise ValueError("Require an unassisted free object with positive mass")
    if any(t != mujoco.mjtTrn.mjTRN_JOINT for t in model.actuator_trntype):
        raise ValueError("This pilot supports explicit scalar joint actuators only")
    if any(model.jnt_bodyid[j] == obj for j in model.actuator_trnid[:, 0]):
        raise ValueError("The object must not have actuators")


def command(model, data, q):
    for i, joint in enumerate(model.actuator_trnid[:, 0]):
        data.ctrl[i] = q[model.jnt_qposadr[joint]]


def object_transform(model, data):
    import mujoco

    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task_object")
    result = np.eye(4)
    result[:3, :3] = data.xmat[obj].reshape(3, 3)
    result[:3, 3] = data.xpos[obj]
    return result


def set_wrist_frame(model, data, config, position, rotation):
    import mujoco

    root = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, config["robot"]["root_body"]
    )
    rotation = np.asarray(rotation)
    model.body_pos[root] = position
    model.body_quat[root] = quaternion(rotation)
    for axis, vector in zip("xyz", np.eye(3)):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fixture_" + axis)
        model.jnt_axis[joint] = rotation.T @ vector
    mujoco.mj_setConst(model, data)


def settle(config):
    import mujoco

    model, data, mesh = load_scene(config)
    mujoco.mj_forward(model, data)
    initial = object_transform(model, data)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_free")
    qid, vid = model.jnt_qposadr[obj], model.jnt_dofadr[obj]
    velocities = []
    for _ in range(round(config["rollout"]["settle_s"] / model.opt.timestep)):
        command(model, data, model.qpos0)
        mujoco.mj_step(model, data)
        if data.time >= config["rollout"]["settle_s"] - 0.2:
            velocities.append(data.qvel[vid : vid + 6].copy())
    mujoco.mj_forward(model, data)
    final = object_transform(model, data)
    clearance = float(trimesh.transform_points(mesh.vertices, final)[:, 2].min())
    v = np.asarray(velocities)
    report = dict(
        initial_world_from_object=initial.tolist(),
        settled_world_from_object=final.tolist(),
        translation_m=float(np.linalg.norm(final[:3, 3] - initial[:3, 3])),
        rotation_deg=float(
            np.rad2deg(
                Rotation.from_matrix(final[:3, :3] @ initial[:3, :3].T).magnitude()
            )
        ),
        minimum_height_m=clearance,
        last_linear_speed_max_m_s=float(np.linalg.norm(v[:, :3], axis=1).max()),
        last_angular_speed_max_rad_s=float(np.linalg.norm(v[:, 3:], axis=1).max()),
        object_qpos=data.qpos[qid : qid + 7].tolist(),
    )
    mode = config.get("initialization", "reconstruction_preserving")
    if mode not in ("reconstruction_preserving", "gravity_settled_task"):
        raise ValueError("Unknown task initialization mode")
    report["source_pose_preserved_within_2cm"] = report["translation_m"] < 0.02
    report["initialization"] = mode
    report["accepted"] = (
        report["translation_m"]
        < (0.02 if mode == "reconstruction_preserving" else 0.05)
        and report["rotation_deg"] < 15
        and abs(clearance) < 0.004
        and report["last_linear_speed_max_m_s"] < 0.01
        and report["last_angular_speed_max_rad_s"] < 0.15
    )
    write_json(Path(config["output"]) / "settling.json", report)
    return report


def plan(config):
    import mujoco
    from real2sim2real.retargeting.object_grasp import surface_pairs
    from real2sim2real.retargeting.contact_layout import (
        surface_layouts,
        contact_targets,
    )
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        fit_contacts,
        contact_penetrations,
    )

    output = Path(config["output"]) / config.get("planning_directory", "planning")
    output.mkdir(exist_ok=False)
    settled = settle(config)
    if not settled["accepted"]:
        raise ValueError(
            "Object settling failed; do not plan through an unstable scene"
        )
    model, data, mesh = load_scene(config)
    world = mesh.copy()
    world.apply_transform(settled["settled_world_from_object"])
    settings = config["planning"]
    method = settings.get("surface_search", "outer")
    if method not in ("outer", "local_spans"):
        raise ValueError("Unknown surface search method")
    generator = surface_layouts if method == "local_spans" else surface_pairs
    search_diagnostics = {}
    search_options = (
        dict(
            yaw_samples=settings.get("yaw_samples", 8),
            gravity_lever_weight=settings.get("gravity_lever_weight", 1.0),
            diagnostics=search_diagnostics,
        )
        if method == "local_spans"
        else {}
    )
    pairs = generator(
        world,
        min_width_m=settings["min_width_m"],
        max_width_m=settings["max_width_m"],
        minimum_height_m=settings["minimum_height_m"],
        max_candidates=max(1, settings["max_candidates"] // 2),
        cues=config.get("human_cues", []),
        finger_offsets_m=settings.get("finger_offsets_m", [-0.012, 0.012]),
        **search_options,
    )
    write_json(
        output / "surface_pairs.json",
        dict(
            pairs=pairs,
            cues=config.get("human_cues", []),
            human_finger_targets_used=False,
            search_method=method,
            search_diagnostics=search_diagnostics,
        ),
    )
    idx = robot_indices(model, config["robot"]["root_body"])
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_free"
    )
    object_qid = model.jnt_qposadr[object_joint]
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
        for j in idx["joints"]
    ]
    z_index = names.index("fixture_z")
    rows = []
    target_rejections = []
    for pair in pairs:
        for direction in [1, -1]:
            offsets = settings.get("finger_offsets_m", [-0.012, 0.012])
            if settings.get("orient_finger_order", False) and direction < 0:
                offsets = offsets[::-1]
            # Reversing the wrist must also exchange thumb/opposing surfaces.
            targets = contact_targets(world, pair, offsets, direction=direction)
            if targets is None:
                target_rejections.append(
                    dict(
                        pair=pair,
                        direction=direction,
                        reason="missing_opposing_surface",
                    )
                )
                continue
            number = len(rows)
            close = np.asarray(pair["close_axis"]) * direction
            x = np.array([0.0, 0.0, -1.0])
            z = close
            y = np.cross(z, x)
            rotation = np.stack([x, y, z], axis=1)
            position = np.asarray(pair["center_m"]) - rotation @ np.array(
                [0.055, 0.01, 0.13]
            )
            set_wrist_frame(model, data, config, position, rotation)
            mujoco.mj_resetData(model, data)
            data.qpos[object_qid : object_qid + 7] = settled["object_qpos"]
            seed = np.zeros(len(names))
            for i, name in enumerate(names):
                if name.endswith(("MCP_FE", "PIP", "DIP", "IP")):
                    seed[i] = 0.7
                if name.endswith("thumb_CMC_FE"):
                    seed[i] = 1.2
            fit = fit_contacts(
                model,
                data,
                config["robot"]["root_body"],
                config["robot"]["sites"],
                targets,
                seed,
                max_nfev=settings["max_nfev"],
            )
            q = fit["q"]
            opened = q.copy()
            for i, name in enumerate(names):
                if name.endswith(("MCP_FE", "PIP", "DIP", "IP", "thumb_CMC_FE")):
                    opened[i] -= 0.25
            limits = model.jnt_range[idx["joints"]]
            opened = np.clip(opened, limits[:, 0], limits[:, 1])
            pre = opened.copy()
            pre[z_index] += config["rollout"]["approach_offset_m"]
            approach_penetration = 0.0
            for fraction in np.linspace(0, 1, 15):
                data.qpos[idx["q"]] = (1 - fraction) * pre + fraction * opened
                mujoco.mj_forward(model, data)
                approach_penetration = max(
                    approach_penetration,
                    float(contact_penetrations(model, data, idx).max(initial=0)),
                )
            reasons = []
            if fit["site_errors_m"].max() > settings["contact_error_m"]:
                reasons.append("contact_target_error")
            if fit["penetration_m"].max(initial=0) > settings["max_penetration_m"]:
                reasons.append("grasp_penetration")
            if approach_penetration > settings["max_penetration_m"]:
                reasons.append("approach_penetration")
            row = dict(
                candidate_id=number,
                thumb_side_direction=direction,
                pair=pair,
                targets_m=targets.tolist(),
                base_position_m=position.tolist(),
                base_rotation=rotation.tolist(),
                joint_names=names,
                q_grasp=q.tolist(),
                q_open=opened.tolist(),
                q_pregrasp=pre.tolist(),
                site_errors_m=fit["site_errors_m"].tolist(),
                max_penetration_m=float(fit["penetration_m"].max(initial=0)),
                approach_max_penetration_m=approach_penetration,
                nfev=fit["nfev"],
                converged=fit["converged"],
                accepted=not reasons,
                rejected_for=reasons,
                score=float(
                    fit["site_errors_m"].max()
                    + fit["penetration_m"].max(initial=0)
                    + pair["score"] * 0.01
                ),
            )
            rows.append(row)
            print(
                f"Grasp {number}: accepted={row['accepted']} error={max(row['site_errors_m']) * 1000:.1f} mm approach={approach_penetration * 1000:.1f} mm",
                flush=True,
            )
            write_json(output / "progress.json", dict(candidates=rows))
    report = dict(
        schema="real2sim2real.object-grasp-plan/1",
        candidates=rows,
        accepted=sum(r["accepted"] for r in rows),
        settling=settled,
        scene_receipt_sha256=digest(Path(config["output"]) / "scene/receipt.json"),
        human_finger_reference_used=False,
        physical_success_validated=False,
        target_rejections=target_rejections,
        surface_search=method,
    )
    write_json(output / "plan.json", report)
    return report


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def refine_approaches(config):
    """Reuse checked contact IK; search bounded translated opening paths."""
    import mujoco
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        contact_penetrations,
    )

    root = Path(config["output"])
    parent = Path(config["approach_refinement"]["parent_plan"])
    if digest(parent) != config["approach_refinement"]["parent_sha256"]:
        raise ValueError("Contact plan changed")
    report = json.loads(parent.read_text())
    if report["scene_receipt_sha256"] != digest(root / "scene/receipt.json"):
        raise ValueError("Contact plan belongs to another scene")
    output = root / config["planning_directory"]
    output.mkdir(exist_ok=False)
    for candidate in report["candidates"]:
        model, data, _ = load_scene(config)
        set_wrist_frame(
            model,
            data,
            config,
            candidate["base_position_m"],
            candidate["base_rotation"],
        )
        idx = robot_indices(model, config["robot"]["root_body"])
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_free")
        qid = model.jnt_qposadr[joint]
        data.qpos[qid : qid + 7] = report["settling"]["object_qpos"]
        data.qpos[idx["q"]] = candidate["q_grasp"]
        mujoco.mj_forward(model, data)
        sites = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in config["robot"]["sites"]
        ]
        errors = np.linalg.norm(
            data.site_xpos[sites]
            - candidate.get("targets_m", candidate["pair"]["targets_m"]),
            axis=1,
        )
        penetration = float(contact_penetrations(model, data, idx).max(initial=0))
        if not np.allclose(
            errors, candidate["site_errors_m"], atol=1e-8
        ) or not np.isclose(penetration, candidate["max_penetration_m"], atol=1e-8):
            raise ValueError("Saved grasp does not reproduce its geometric checks")
        candidate["rejected_for"] = [
            r for r in candidate["rejected_for"] if r != "approach_penetration"
        ]
        if candidate["rejected_for"]:
            continue
        xyz = [candidate["joint_names"].index("fixture_" + a) for a in "xyz"]
        along, close = (
            np.asarray(candidate["pair"]["along_axis"]),
            np.asarray(candidate["pair"]["close_axis"]),
        )
        up = np.array([0.0, 0.0, 1.0])
        directions = [
            up,
            -along,
            along,
            -close,
            close,
            -along + 0.4 * up,
            along + 0.4 * up,
            -close + 0.4 * up,
            close + 0.4 * up,
        ]
        paths = []
        for vector in directions:
            delta = (
                vector / np.linalg.norm(vector) * config["rollout"]["approach_offset_m"]
            )
            opened = np.asarray(candidate["q_open"])
            pre = opened.copy()
            pre[xyz] += delta
            maximum = 0.0
            samples = int(np.ceil(np.linalg.norm(delta) / 0.002)) + 1
            for a in np.linspace(0, 1, samples):
                data.qpos[idx["q"]] = (1 - a) * pre + a * opened
                mujoco.mj_forward(model, data)
                maximum = max(
                    maximum,
                    float(contact_penetrations(model, data, idx).max(initial=0)),
                )
            paths.append(
                dict(
                    offset_m=delta.tolist(),
                    max_penetration_m=maximum,
                    samples=samples,
                    q_pregrasp=pre.tolist(),
                )
            )
        allowed = [
            p
            for p in paths
            if p["max_penetration_m"] <= config["planning"]["max_penetration_m"]
        ]
        candidate["approach_search"] = paths
        candidate["accepted"] = bool(allowed)
        if allowed:
            chosen = allowed[0]
            candidate["q_pregrasp"] = chosen["q_pregrasp"]
            candidate["approach_max_penetration_m"] = chosen["max_penetration_m"]
            candidate["approach_offset_m"] = chosen["offset_m"]
        else:
            candidate["rejected_for"].append("approach_penetration")
        print(
            f"Approach candidate {candidate['candidate_id']}: {len(allowed)}/{len(paths)} feasible",
            flush=True,
        )
    report.update(
        accepted=sum(r["accepted"] for r in report["candidates"]),
        parent_plan_sha256=digest(parent),
        contact_ik_recomputed=False,
    )
    write_json(output / "plan.json", report)
    return report


def target_at(time, pre, opened, closed, z_index, settings):
    approach, close, lift = (
        settings["approach_s"],
        settings["close_s"],
        settings["lift_s"],
    )
    if time < approach:
        a = smoothstep(time / approach)
        return (1 - a) * pre + a * opened, "approach"
    if time < approach + close:
        a = smoothstep((time - approach) / close)
        return (1 - a) * opened + a * closed, "close"
    result = closed.copy()
    result[z_index] += settings["lift_m"] * smoothstep((time - approach - close) / lift)
    return result, "lift" if time < approach + close + lift else "hold"


def clearance_feedback(displacement, clearance, target, dt, settings):
    """Bounded robot wrist command; never changes the object's simulated pose."""
    speed = np.clip(
        settings["gain_s_inv"] * (target - clearance),
        -settings["max_lower_speed_m_s"],
        settings["max_raise_speed_m_s"],
    )
    return float(np.clip(displacement + dt * speed, 0, settings["max_wrist_lift_m"]))


def load_plan(config):
    """Reject stale geometry or initialization before executing a saved plan."""
    root = Path(config["output"])
    path = root / config.get("planning_directory", "planning") / "plan.json"
    plan = json.loads(path.read_text())
    if plan["scene_receipt_sha256"] != digest(root / "scene/receipt.json"):
        raise ValueError("Grasp plan belongs to another scene")
    settled = json.loads((root / "settling.json").read_text())
    if not settled["accepted"] or settled != plan["settling"]:
        raise ValueError("Task initialization differs from the accepted grasp plan")
    return plan


def verify_selected_trial(evaluation, selected, plan):
    """Bind the selection to its evaluation summary and retained contact plan."""
    rows = [
        r
        for r in evaluation["trials"]
        if r["directory"] == evaluation["selected_trial"]
    ]
    if len(rows) != 1 or not selected["success"]:
        raise ValueError("A unique successful selected trial is required")
    if any(selected.get(k) != v for k, v in rows[0].items() if k != "directory"):
        raise ValueError("Selected trial differs from its pinned evaluation")
    candidate = selected["candidate"]
    if candidate not in plan["candidates"] or not candidate["accepted"]:
        raise ValueError("Selected grasp differs from its accepted contact plan")


def trial(config, candidate, output, *, mass_scale=1.0, friction=None):
    import mujoco
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError("Mass scale must be finite and positive")
    if friction is not None and (not np.isfinite(friction) or friction < 0):
        raise ValueError("Friction must be finite and nonnegative")
    planning = load_plan(config)
    if candidate not in planning["candidates"] or not candidate["accepted"]:
        raise ValueError("Trial requires an unchanged accepted grasp plan")
    model, data, mesh = load_scene(config)
    set_wrist_frame(
        model, data, config, candidate["base_position_m"], candidate["base_rotation"]
    )
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task_object")
    model.body_mass[obj] *= mass_scale
    model.body_inertia[obj] *= mass_scale
    if friction is not None:
        model.geom_friction[:, 0] = friction
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    idx = robot_indices(model, config["robot"]["root_body"])
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_free")
    qid = model.jnt_qposadr[joint]
    settled = json.loads((Path(config["output"]) / "settling.json").read_text())
    data.qpos[qid : qid + 7] = settled["object_qpos"]
    pre, opened, closed = [
        np.asarray(candidate[k]).copy() for k in ["q_pregrasp", "q_open", "q_grasp"]
    ]
    for i, name in enumerate(candidate["joint_names"]):
        if name.endswith(("MCP_FE", "PIP", "DIP", "IP", "thumb_CMC_FE")):
            closed[i] += config["rollout"]["close_offset_rad"]
    limits = model.jnt_range[idx["joints"]]
    closed = np.clip(closed, limits[:, 0], limits[:, 1])
    data.qpos[idx["q"]] = pre
    z_index = candidate["joint_names"].index("fixture_z")
    settings = config["rollout"]
    task = json.loads(Path(config["task"]).read_text())
    table_top = task["table"]["top_z_m"]
    goal = task["goal"]
    if settings["hold_s"] < goal["hold_s"]:
        raise ValueError("Trial hold cannot be shorter than the task requirement")
    feedback = config.get("clearance_feedback")
    contact_feedback = config.get("contact_feedback")
    rotation_feedback = config.get("orientation_feedback")
    closure_correction = 0.0
    rotation_correction = np.zeros(3)
    previous_opposing_force = 0.0
    opposing_indices = [
        i
        for i, name in enumerate(candidate["joint_names"])
        if "thumb" not in name and name.endswith(("MCP_FE", "PIP", "DIP", "IP"))
    ]
    rotation_indices = [candidate["joint_names"].index("fixture_r" + a) for a in "xyz"]
    displacement = 0.0
    duration = sum(settings[k] for k in ["approach_s", "close_s", "lift_s", "hold_s"])
    records = []
    qstates = []
    controls = []
    max_pen = 0.0
    max_table_pen = 0.0
    peak_forces = np.zeros(model.nu)
    aborted = None
    initial_rotation = np.asarray(settled["settled_world_from_object"])[:3, :3]
    object_geoms = set(np.flatnonzero(model.geom_bodyid == obj))
    hand_geoms = set(idx["geoms"])
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    sample_every = max(1, round(0.02 / model.opt.timestep))
    for step in range(round(duration / model.opt.timestep)):
        q, phase = target_at(data.time, pre, opened, closed, z_index, settings)
        if feedback and phase in ("lift", "hold"):
            pose = object_transform(model, data)
            clearance = float(
                (mesh.vertices @ pose[2, :3]).min() + pose[2, 3] - table_top
            )
            displacement = clearance_feedback(
                displacement, clearance, goal["lift_m"], model.opt.timestep, feedback
            )
            q[z_index] = closed[z_index] + displacement
        if contact_feedback and phase in ("lift", "hold"):
            from .grasp_feedback import contact_closure

            closure_correction = contact_closure(
                closure_correction,
                previous_opposing_force,
                model.opt.timestep,
                contact_feedback,
            )
            q[opposing_indices] += closure_correction
        if rotation_feedback and phase in ("lift", "hold"):
            from .grasp_feedback import orientation_correction

            jac = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, None, jac, idx["root"])
            rotation_correction = orientation_correction(
                rotation_correction,
                object_transform(model, data)[:3, :3],
                initial_rotation,
                jac[:, idx["dofs"][rotation_indices]],
                model.opt.timestep,
                rotation_feedback,
            )
            q[rotation_indices] += rotation_correction
        q = np.clip(q, limits[:, 0], limits[:, 1])
        target = data.qpos.copy()
        target[idx["q"]] = q
        command(model, data, target)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            aborted = "nonfinite_state"
            break
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError("External assistance applied")
        thumb = fingers = 0.0
        group_forces = dict.fromkeys(
            ("thumb", "index", "middle", "ring", "pinky", "palm_or_other"), 0.0
        )
        table_contacts = 0
        for contact_id, c in enumerate(data.contact):
            pair = {c.geom1, c.geom2}
            if pair & hand_geoms:
                max_pen = max(max_pen, max(0.0, -float(c.dist)))
                if table in pair:
                    max_table_pen = max(max_table_pen, max(0.0, -float(c.dist)))
            if pair & object_geoms and table in pair:
                table_contacts += 1
            if pair & object_geoms and pair & hand_geoms:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_id, force)
                gid = next(iter(pair & hand_geoms))
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
                )
                group = next(
                    (g for g in group_forces if g != "palm_or_other" and g in name),
                    "palm_or_other",
                )
                group_forces[group] += max(0.0, float(force[0]))
                if "thumb" in name:
                    thumb += max(0.0, float(force[0]))
                elif any(x in name for x in ["index", "middle", "ring", "pinky"]):
                    fingers += max(0.0, float(force[0]))
        previous_opposing_force = fingers
        peak_forces = np.maximum(peak_forces, np.abs(data.actuator_force))
        if step % sample_every == 0 or step == round(duration / model.opt.timestep) - 1:
            mujoco.mj_forward(model, data)
            transform = object_transform(model, data)
            height = float(
                trimesh.transform_points(mesh.vertices, transform)[:, 2].min()
            )
            tilt = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        transform[:3, :3] @ initial_rotation.T
                    ).magnitude()
                )
            )
            records.append(
                dict(
                    time_s=float(data.time),
                    phase=phase,
                    minimum_object_height_m=height,
                    object_position_m=transform[:3, 3].tolist(),
                    wrist_position_m=data.xpos[idx["root"]].tolist(),
                    wrist_rotation=data.xmat[idx["root"]].reshape(3, 3).tolist(),
                    rotation_change_deg=tilt,
                    thumb_force_n=thumb,
                    finger_force_n=fingers,
                    table_contacts=table_contacts,
                    normal_contact_force_by_group_n=group_forces,
                    opposing_closure_correction_rad=closure_correction,
                    wrist_rotation_correction_rad=rotation_correction.tolist(),
                )
            )
            qstates.append(data.qpos.copy())
            controls.append(data.ctrl.copy())
            if (
                height < -0.04
                or np.linalg.norm(
                    data.qvel[model.jnt_dofadr[joint] : model.jnt_dofadr[joint] + 3]
                )
                > 5
            ):
                aborted = "object_dropped_or_unstable"
                break
    from .grasp_hold import measure_hold, hold_rejections

    metrics = measure_hold(records, table_top_m=table_top)
    failures = hold_rejections(
        metrics,
        minimum_clearance_m=goal["minimum_clearance_m"],
        observed_max_penetration_m=max_pen,
        allowed_max_penetration_m=config["planning"]["max_penetration_m"],
        aborted=aborted,
    )
    output = Path(output)
    output.mkdir(exist_ok=False)
    np.savez_compressed(
        output / "states.npz",
        qpos=np.asarray(qstates),
        ctrl=np.asarray(controls),
        timestamps_s=np.array([r["time_s"] for r in records]),
    )
    report = dict(
        schema="real2sim2real.object-grasp-trial/2",
        candidate_id=candidate["candidate_id"],
        candidate=candidate,
        mass_scale=mass_scale,
        friction=config["physics"]["friction"] if friction is None else friction,
        success=not failures,
        rejected_for=failures,
        **metrics,
        hold_slip_frame="wrist",
        max_robot_penetration_m=max_pen,
        max_robot_table_penetration_m=max_table_pen,
        actuator_peak_forces={
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): dict(
                peak_abs=float(value),
                units="N"
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                in ("fixture_x", "fixture_y", "fixture_z")
                else "Nm",
                limit=model.actuator_forcerange[i].tolist(),
            )
            for i, value in enumerate(peak_forces)
        },
        controller=dict(
            clearance_feedback=feedback,
            contact_feedback=contact_feedback,
            orientation_feedback=rotation_feedback,
            close_offset_rad=settings["close_offset_rad"],
            duration_s=duration,
        ),
        configuration=config,
        records=records,
        source_scene_sha256=digest(Path(config["output"]) / "scene/scene.xml"),
        states_sha256=digest(output / "states.npz"),
        mujoco_version=mujoco.__version__,
        object_assistance=False,
        arm_reachability_validated=False,
        physical_hardware_validated=False,
    )
    write_json(output / "result.json", report)
    return report


def evaluate(config):
    root = Path(config["output"])
    output = root / config.get("evaluation_directory", "evaluation")
    output.mkdir(exist_ok=False)
    planning = load_plan(config)
    candidates = sorted(
        [r for r in planning["candidates"] if r["accepted"]], key=lambda r: r["score"]
    )[:6]
    rows = []
    closures = config.get("closure_search_rad", [config["rollout"]["close_offset_rad"]])
    if not 1 <= len(closures) <= 3 or any(not 0 <= c <= 0.5 for c in closures):
        raise ValueError("Use one to three bounded closure offsets")
    for candidate in candidates:
        for closure_id, closure in enumerate(closures):
            settings = deepcopy(config)
            settings["rollout"]["close_offset_rad"] = closure
            name = f"candidate-{candidate['candidate_id']}-closure-{closure_id}"
            result = trial(settings, candidate, output / name)
            rows.append(
                dict(
                    directory=name,
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in ["records", "candidate"]
                    },
                )
            )
            print(
                f"Dynamic {name}: {result['success']} {result['rejected_for']}",
                flush=True,
            )
    passed = [r for r in rows if r["success"]]
    chosen = min(passed, key=lambda r: r["hold_slip_m"]) if passed else None
    selected = chosen["candidate_id"] if chosen else None
    variants = []
    if selected is not None:
        candidate = next(c for c in candidates if c["candidate_id"] == selected)
        settings = deepcopy(config)
        settings["rollout"]["close_offset_rad"] = chosen["controller"][
            "close_offset_rad"
        ]
        for name, mass, friction in [
            ("repeat", 1.0, config["physics"]["friction"]),
            ("heavy-low-friction", 2.0, 0.4),
            ("light-low-friction", 0.5, 0.4),
        ]:
            result = trial(
                settings, candidate, output / name, mass_scale=mass, friction=friction
            )
            variants.append(
                dict(
                    directory=name,
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in ["records", "candidate"]
                    },
                )
            )
    report = dict(
        schema="real2sim2real.object-grasp-evaluation/1",
        trials=rows,
        selected_candidate=selected,
        selected_trial=None if chosen is None else chosen["directory"],
        variants=variants,
        decision="ready_for_arm_reachability_study"
        if selected is not None and all(v["success"] for v in variants)
        else "nominal_grasp_only"
        if selected is not None
        else "no_validated_dynamic_grasp",
        human_finger_targets_used=False,
        object_assistance=False,
        full_episode_control_validated=False,
    )
    write_json(output / "evaluation.json", report)
    return report


def verify_frozen_controller(config, selected):
    """Reject feedback or trajectory retuning disguised as fixed-grasp validation."""
    for key in ("clearance_feedback", "contact_feedback", "orientation_feedback"):
        if config.get(key) != selected["controller"].get(key):
            raise ValueError("Frozen validation cannot change controller feedback")
    original = selected.get("configuration", {}).get("rollout")
    if original is not None:
        current = config["rollout"]
        # Closure is restored from the selected trial by validate_selected.
        excluded = {"hold_s", "close_offset_rad"}
        if {k: v for k, v in current.items() if k not in excluded} != {
            k: v for k, v in original.items() if k not in excluded
        }:
            raise ValueError(
                "Frozen validation cannot change the approach/lift trajectory"
            )
        if not np.isfinite(current["hold_s"]) or current["hold_s"] < original["hold_s"]:
            raise ValueError("Frozen validation cannot shorten the selected hold")


def validate_selected(config):
    """Evaluate one frozen selection on declared physical-parameter cases."""
    root = Path(config["output"])
    parent = Path(config["validation"]["parent_evaluation"])
    if digest(parent) != config["validation"]["parent_sha256"]:
        raise ValueError("Selected evaluation changed")
    evaluation = json.loads(parent.read_text())
    if evaluation["selected_trial"] is None:
        raise ValueError("A nominally successful frozen candidate is required")
    selected = json.loads(
        (parent.parent / evaluation["selected_trial"] / "result.json").read_text()
    )
    verify_selected_trial(evaluation, selected, load_plan(config))
    verify_frozen_controller(config, selected)
    if (
        digest(parent.parent / evaluation["selected_trial"] / "states.npz")
        != selected["states_sha256"]
    ):
        raise ValueError("Selected trial states changed")
    settings = deepcopy(config)
    settings["rollout"]["close_offset_rad"] = selected["controller"]["close_offset_rad"]
    output = root / config["evaluation_directory"]
    output.mkdir(exist_ok=False)
    # Copy only compact summaries; retained recordings remain at their originals.
    evaluation["trials"] = []
    evaluation["variants"] = []
    rows = []
    cases = config["validation"]["cases"]
    if not 1 <= len(cases) <= 12:
        raise ValueError("Bounded validation case list required")
    for case in cases:
        name = case["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Use a local case name")
        result = trial(
            settings,
            selected["candidate"],
            output / name,
            mass_scale=case["mass_scale"],
            friction=case["friction"],
        )
        rows.append(
            dict(
                directory=name,
                **{
                    k: v for k, v in result.items() if k not in ("records", "candidate")
                },
            )
        )
        print(
            f"Frozen validation {name}: {result['success']} {result['rejected_for']}",
            flush=True,
        )
    evaluation.update(
        trials=rows,
        selected_trial=None,
        variants=[],
        parent_evaluation_sha256=digest(parent),
        fixed_candidate_id=selected["candidate_id"],
        fixed_closure_offset_rad=settings["rollout"]["close_offset_rad"],
        decision="passes_declared_parameter_cases"
        if all(r["success"] for r in rows)
        else "nominal_success_not_robust_to_declared_parameters",
    )
    write_json(output / "evaluation.json", evaluation)
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["prepare", "plan", "approach", "evaluate", "validate", "review"],
        required=True,
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--phase",
                args.phase,
                "--worker",
            ],
            timeout=config["deadlines_s"][args.phase],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide GPU for the CPU grasp pilot")
    if args.phase == "prepare":
        from .object_grasp_scene import prepare

        return prepare(config)
    if args.phase == "plan":
        return plan(config)
    if args.phase == "approach":
        return refine_approaches(config)
    if args.phase == "evaluate":
        return evaluate(config)
    if args.phase == "validate":
        return validate_selected(config)
    from .object_grasp_review import review

    return review(config)


if __name__ == "__main__":
    main()
