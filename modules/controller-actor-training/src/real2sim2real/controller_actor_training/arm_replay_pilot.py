"""Bounded two-arm joint control of a screened saved grasp trajectory on CPU.

Finger commands replay the source trial; arm motors use bounded PD plus model
bias compensation. Optional bounded feedback uses privileged object poses and
contact forces. The object remains free in every condition.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .arm_path_audit import place_bases, joint_neighbors
from .object_grasp_pilot import load_scene, object_transform, verify_unassisted_model
from .grasp_hold import measure_hold, hold_rejections
from .support_control import (
    contact_wrench_on_body,
    support_hold_metrics,
    support_rejections,
)
from .arm_object_feedback import (
    ArmObjectFeedback,
    joint_offset,
    support_contact_evidence,
    SupportHeightAlignment,
)
from .arm_replay_diagnostics import configure_contacts, wrist_errors, phase_maxima
from .arm_tracking_control import ArmTrackingCompensator, InertialFeedforward
from .arm_robustness import validate_case, apply_case
from .reference_extension import effective_parent
from .simulation_health import step_rejection
from .arm_motion_condition import apply_condition


def controller_clock(physics_step_s, requested_period_s=None):
    """Require an exact bounded number of physics steps per control update."""
    period = physics_step_s if requested_period_s is None else requested_period_s
    if (
        isinstance(period, bool)
        or not np.isfinite([physics_step_s, period]).all()
        or physics_step_s <= 0
        or not physics_step_s <= period <= 0.01
        or not np.isclose(
            period / physics_step_s, round(period / physics_step_s), atol=1e-9, rtol=0
        )
    ):
        raise ValueError(
            "Control period must be an integer physics-step multiple up to 10 ms"
        )
    return float(period), int(round(period / physics_step_s))


def motor_command(target, position, velocity, bias, kp, kd, limits):
    """Finite, explicitly torque-bounded joint control, including feedforward."""
    values = [
        np.asarray(v, float) for v in (target, position, velocity, bias, kp, kd, limits)
    ]
    if any(not np.isfinite(v).all() for v in values) or (values[-1] <= 0).any():
        raise ValueError("Finite states and positive torque limits required")
    if (values[-3] < 0).any() or (values[-2] < 0).any():
        raise ValueError("Nonnegative gains required")
    raw = values[4] * (values[0] - values[1]) - values[5] * values[2] + values[3]
    return np.clip(raw, -values[6], values[6]), np.abs(raw) > values[6]


def bias_compensated_target(target, bias, kp, *, target_velocity=0.0, kd=0.0):
    if (
        not np.isfinite(kp)
        or kp <= 0
        or not np.isfinite(target).all()
        or not np.isfinite(bias).all()
        or not np.isfinite(target_velocity).all()
        or not np.isfinite(kd)
        or kd < 0
    ):
        raise ValueError("Finite target/bias and positive stiffness required")
    return (
        np.asarray(target) + (np.asarray(bias) + kd * np.asarray(target_velocity)) / kp
    )


def subtree_geoms(model, root):
    result = set()
    for geom in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom])
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            result.add(geom)
    return result


def rollout(config, interaction=None):
    """Shared simulation coroutine; an optional adapter pauses at policy ticks.

    The adapter can request pressure offsets only. Default execution never yields
    and retains the original controller operations and physical clock.
    """
    import mujoco

    finger_mode = config.get("finger_control_mode", "archived")
    descent_spec = config.get("descent_pad_control")
    if descent_spec is not None:
        from .descent_pad_control import validate_source

        validate_source(descent_spec, config)
        if finger_mode != "live_contact_force" or interaction is not None:
            raise ValueError(
                "Descent trials require the existing live finger owner without an actor"
            )
    closed_spec = config.get("closed_loop_grasp_feedback")
    if closed_spec is not None:
        from .closed_loop_grasp import validate_source

        validate_source(closed_spec, config)
        if (
            finger_mode != "live_contact_force"
            or interaction is not None
            or any(
                config.get(k)
                for k in (
                    "object_feedback",
                    "support_height_alignment",
                    "bounded_grasp_correction",
                )
            )
        ):
            raise ValueError(
                "Closed-loop grasp feedback requires exclusive live arm/finger ownership"
            )
    correction_spec = config.get("bounded_grasp_correction")
    if correction_spec is not None:
        from .bounded_grasp_correction import validate

        validate(correction_spec, config)
        if finger_mode != "live_contact_force" or interaction is not None:
            raise ValueError(
                "Bounded grasp probes require the live finger owner without another residual"
            )
    if finger_mode not in ("archived", "live_contact_force"):
        raise ValueError(
            "Explicit archived or live_contact_force finger control required"
        )
    if finger_mode == "live_contact_force" and config.get("finger_contact_trim"):
        raise ValueError(
            "Live pad feedback and separate finger trim cannot share ownership"
        )
    if config.get("finger_seating_plan") and finger_mode != "live_contact_force":
        raise ValueError("Seating proposals require the live finger owner")
    if "acquisition_profile" in config:
        from .acquisition_profile import validate

        validate(config["acquisition_profile"])
        if finger_mode != "live_contact_force":
            raise ValueError("Acquisition profiles require the live finger owner")
    if config.get("finger_feedback_override") and finger_mode != "live_contact_force":
        raise ValueError("Finger feedback overrides require the live owner")
    simulation_case = validate_case(config)
    stationary_admission = None
    if config.get("finger_plan"):
        if finger_mode != "live_contact_force":
            raise ValueError("Replanned finger ownership requires live feedback")
    if config.get("motion_condition", "demonstrated") == "demonstrated" and (
        config.get("finger_plan")
        or config.get("acquisition_profile")
        or config.get("finger_seating_plan")
        or config.get("finger_feedback_override", {}).get("unloaded_acquisition")
        or config.get("validated_stationary")
        or config.get("diagnostic_stationary")
    ):
        from .arm_finger_plan import validate_stationary

        stationary_admission = validate_stationary(config)
    if config.get("precontact_support_alignment") and not config.get("object_feedback"):
        raise ValueError(
            "Precontact alignment requires explicit bounded object feedback"
        )
    implementation_sources = {
        name: Path(__file__).with_name(name).read_bytes()
        for name in (
            "arm_replay_pilot.py",
            "arm_object_feedback.py",
            "arm_replay_diagnostics.py",
            "arm_tracking_control.py",
            "arm_robustness.py",
            "reference_extension.py",
            "simulation_health.py",
            "finger_contact_trim.py",
            "arm_finger_feedback.py",
            "contact_force_control.py",
            "payload_pressure.py",
            "contact_patch_motion.py",
            "contact_anchor.py",
            "arm_motion_condition.py",
            "arm_finger_plan.py",
            "continuous_reference.py",
            "bounded_grasp_correction.py",
            "closed_loop_grasp.py",
            "slip_tilt_feedback.py",
            "descent_pad_control.py",
            "finger_release.py",
            "finger_release_plan.py",
            "wrist_withdrawal.py",
            "pad_seating_plan.py",
            "finger_acquisition.py",
            "closing_wrist_offsets.py",
            "acquisition_profile.py",
            "arm_clearance_audit.py",
        )
    }
    audit_path = Path(config["audit"])
    if digest(audit_path) != config["audit_sha256"]:
        raise ValueError("Pinned arm audit changed")
    audit = json.loads(audit_path.read_text())
    for source, sha in audit["sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("Arm audit input changed")
    rows = [
        r for r in audit["placements"] if r["placement"]["name"] == config["placement"]
    ]
    if (
        len(rows) != 1
        or not rows[0]["screen_pass"]
        or audit.get("retained_fixture_hands")
    ):
        raise ValueError("A passing complete two-arm path is required")
    row = rows[0]
    arms = audit["config"]["arms"]
    if {a["name"] for a in arms} != {"left", "right"}:
        raise ValueError("Both hands must have arms")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    from .closing_wrist_offsets import nonzero

    if nonzero(parent.get("continuous_protocol", {}).get("closing_wrist_offsets")) and (
        finger_mode != "live_contact_force" or not config.get("finger_seating_plan")
    ):
        raise ValueError("Closing wrist preload requires its coupled live finger plan")
    required_posture = parent.get("continuous_protocol", {}).get(
        "finger_release_posture"
    )
    if required_posture and config.get("finger_release_posture") != required_posture:
        raise ValueError(
            "Live release must use the exact posture in the audited command plan"
        )
    if config.get("finger_release_posture") and (
        finger_mode != "live_contact_force"
        or not parent.get("continuous_protocol", {}).get("release_planned")
    ):
        raise ValueError("A release posture requires continuous live-finger release")
    parent = effective_parent(parent, audit)
    old, _, mesh = load_scene(parent["configuration"])
    archive = Path(audit["config"]["trial"]).with_name("states.npz")
    if digest(archive) != parent["states_sha256"]:
        raise ValueError("Source states changed")
    with np.load(archive) as f:
        saved = {k: f[k].copy() for k in f.files}
    samples = row["samples"]
    if [s["source_index"] for s in samples] != list(range(len(saved["timestamps_s"]))):
        raise ValueError("Screen every archived timestamp before dynamics")
    times = saved["timestamps_s"]
    if (
        len(times) < 2
        or not 0 < times[-1] - times[0] <= 30
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Require a finite pilot of at most 30 seconds")
    source_scene = audit_path.parent / "audit-scene.xml"
    receipt = json.loads((audit_path.parent / "assembly.json").read_text())
    if digest(source_scene) != receipt["scene_sha256"]:
        raise ValueError("Assembled scene changed")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "implementation"
    snapshot.mkdir()
    for name, contents in implementation_sources.items():
        (snapshot / name).write_bytes(contents)
    root = ET.parse(source_scene).getroot()
    actuator = root.find("actuator")
    names = [n for arm in arms for n in arm["joints"]]
    for name in names:
        limit = config["torque_limits_nm"][name]
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Positive motor torque budget required")
        ET.SubElement(
            actuator,
            "position",
            name="pilot__" + name,
            joint=name,
            kp=str(config["kp_nm_per_rad"]),
            kv=str(config["kd_nm_s_per_rad"]),
            forcerange=f"{-limit} {limit}",
        )
    # Let MuJoCo integrate actuator damping implicitly rather than adding an
    # explicit high-gain velocity torque outside its derivative model.
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", "implicitfast")
    if "timestep_s" in config:
        step = float(config["timestep_s"])
        if not 0.0001 <= step <= 0.002:
            raise ValueError("Diagnostic timestep must be between 0.1 and 2 ms")
        option.set("timestep", str(step))
    scene = out / "scene.xml"
    ET.ElementTree(root).write(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    verify_unassisted_model(model)
    if any("fixture_" in (model.joint(j).name or "") for j in range(model.njnt)):
        raise ValueError("No Cartesian wrist fixtures allowed")
    place_bases(model, audit["config"], row["placement"])
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_saveLastXML(str(scene), model)
    mujoco.mj_resetData(model, data)
    qids = np.array([model.joint(n).qposadr[0] for n in names])
    vids = np.array([model.joint(n).dofadr[0] for n in names])
    aids = np.array([model.actuator("pilot__" + n).id for n in names])
    limits = np.array([config["torque_limits_nm"][n] for n in names])
    speed_limits = np.array(
        [audit["config"]["velocity_limits_rad_s"][n] for n in names]
    )
    ranges = model.jnt_range[[model.joint(n).id for n in names]]
    if not np.isfinite(speed_limits).all() or (speed_limits <= 0).any():
        raise ValueError("Positive arm velocity limits required")
    targets = np.array(
        [np.concatenate([s["arms"][a["name"]]["q"] for a in arms]) for s in samples]
    )
    motion_condition = config.get("motion_condition", "demonstrated")
    parent, targets = apply_condition(parent, targets, motion_condition)
    acquisition_path = None
    if config.get("acquisition_profile"):
        from .acquisition_profile import arm_targets, screen

        targets = arm_targets(
            times,
            targets,
            arms,
            parent["configuration"]["rollout"],
            config["acquisition_profile"],
        )
        acquisition_path = screen(
            model,
            qids,
            targets,
            times,
            speed_limits,
            audit["config"]["criteria"]["joint_margin_rad"],
        )
        write_json(out / "acquisition-path.json", acquisition_path)
        if not acquisition_path["accepted"]:
            raise ValueError(
                "Acquisition arm schedule fails native speed/margin admission"
            )
    accelerations = np.gradient(np.gradient(targets, times, axis=0), times, axis=0)
    inertia_control = (
        InertialFeedforward(config["inertial_feedforward"], limits)
        if config.get("inertial_feedforward")
        else None
    )
    mass_matrix = np.zeros((model.nv, model.nv))
    data.qpos[qids] = targets[0]
    for j in range(old.njnt):
        name = old.joint(j).name
        ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if ident >= 0:
            width = 7 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            src, dst = old.jnt_qposadr[j], model.jnt_qposadr[ident]
            data.qpos[dst : dst + width] = saved["qpos"][0, src : src + width]
    finger_pairs = [
        (old.actuator(model.actuator(i).name).id, i)
        for i in range(model.nu)
        if i not in aids
    ]
    data.time = float(times[0])
    mujoco.mj_forward(model, data)
    initial_rotation = object_transform(model, data)[:3, :3].copy()
    obj = model.body("task_object").id
    table = model.geom("table").id
    roots = {a["name"]: model.body(a["root"]).id for a in arms}
    hands = {name: subtree_geoms(model, body) for name, body in roots.items()}
    object_geoms = subtree_geoms(model, obj)
    trim, trim_ids = None, None
    if config.get("finger_contact_trim"):
        from .finger_contact_trim import bind

        trim, trim_ids = bind(
            model,
            config["finger_contact_trim"],
            [dst for _, dst in finger_pairs],
            roots["right"],
        )
    arm_bodies = {model.body(n).id for a in arms for n in a["collision_bodies"]}
    arm_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies
    }
    robot_geoms = arm_geoms | hands["left"] | hands["right"]
    condition = config.get("contact_condition", "loaded")
    contact_exclusions = configure_contacts(
        root,
        model,
        condition=condition,
        arm_bodies=arm_bodies,
        hand_geoms=hands["left"] | hands["right"],
        object_geoms=object_geoms,
        exclude_adjacent=config.get("exclude_adjacent_arm_collisions", False),
    )
    if condition != "loaded" or config.get("exclude_adjacent_arm_collisions", False):
        # Recompile once; preserve the placed bases and initial state. IDs must
        # remain unchanged because only contact exclusions have been added.
        initial_q = data.qpos.copy()
        ET.ElementTree(root).write(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        place_bases(model, audit["config"], row["placement"])
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        mujoco.mj_saveLastXML(str(scene), model)
        data.qpos[:] = initial_q
        data.time = float(times[0])
        mujoco.mj_forward(model, data)
    if simulation_case is not None:
        apply_case(model, data, simulation_case)
        mujoco.mj_saveLastXML(str(scene), model)
    live_fingers = None
    finger_plan_admission = None
    if finger_mode == "live_contact_force":
        from .arm_finger_feedback import ArmFingerFeedback

        if condition != "loaded":
            raise ValueError("Live contact feedback requires loaded contacts")
        if config.get("finger_plan"):
            from .arm_finger_plan import bind

            parent, finger_plan_admission = bind(parent, config["finger_plan"], out)
        live_fingers = ArmFingerFeedback(
            model,
            data,
            parent,
            roots,
            mesh,
            [dst for _, dst in finger_pairs],
            overrides=config.get("finger_feedback_override"),
        )
        if config.get("finger_release_posture"):
            from .finger_release_plan import bind

            bind(
                live_fingers,
                model,
                config["finger_release_posture"],
                withdrawal=parent.get("continuous_protocol", {}).get(
                    "wrist_withdrawal"
                ),
                release=parent.get("continuous_protocol", {}).get("finger_release"),
            )
        if config.get("finger_seating_plan"):
            from .pad_seating_plan import bind

            bind(
                live_fingers,
                model,
                config["finger_seating_plan"],
                closing_wrist_offsets=parent.get("continuous_protocol", {}).get(
                    "closing_wrist_offsets"
                ),
            )
    finger_path = None
    descent = None
    if descent_spec is not None:
        from .descent_pad_control import bind

        descent = bind(live_fingers, descent_spec, parent)
    if config.get("acquisition_profile"):
        from .acquisition_profile import bind_fingers

        bind_fingers(live_fingers, model, config["acquisition_profile"])
    correction_path = None
    if correction_spec is not None:
        from .bounded_grasp_correction import bind_fingers, wrist_targets

        bind_fingers(
            live_fingers, model, correction_spec, parent["continuous_protocol"]
        )
        targets, correction_path = wrist_targets(
            model,
            data,
            targets,
            times,
            qids,
            vids,
            arms,
            roots,
            parent,
            correction_spec,
            speed_limits,
            audit["config"]["criteria"]["joint_margin_rad"],
        )
        write_json(out / "bounded-correction-path.json", correction_path)
        if not correction_path["accepted"]:
            raise ValueError(
                "Bounded grasp correction fails wrist IK/speed/margin admission"
            )
        accelerations = np.gradient(np.gradient(targets, times, axis=0), times, axis=0)
        np.savez_compressed(
            out / "bounded-command-targets.npz", targets=targets, timestamps_s=times
        )
    if finger_plan_admission or parent.get("continuous_protocol"):
        from .arm_finger_plan import path_screen

        finger_path = path_screen(
            model,
            data,
            targets,
            times,
            qids,
            live_fingers,
            arm_geoms,
            robot_geoms,
            object_geoms,
            dict(
                arm_penetration_m=audit["config"]["criteria"]["arm_penetration_m"],
                robot_penetration_m=parent["configuration"]["planning"][
                    "max_penetration_m"
                ],
            ),
        )
        write_json(out / "finger-path.json", finger_path)
        if not finger_path["accepted"]:
            raise ValueError(
                "New finger commands fail assembled arm/environment path admission"
            )
    contact_motion = diagnostic_data = None
    if config.get("contact_patch_motion"):
        from .contact_patch_motion import ContactPatchMotion

        if live_fingers is None:
            raise ValueError("Named pad telemetry requires live finger binding")
        contact_motion = ContactPatchMotion(
            model,
            obj,
            live_fingers.servo.role_geoms,
            live_fingers.servo.pads,
            config["contact_patch_motion"],
        )
        diagnostic_data = mujoco.MjData(model)
    desired = mujoco.MjData(model)
    feedback = (
        ArmObjectFeedback(config["object_feedback"], list(roots))
        if config.get("object_feedback")
        else None
    )
    if feedback and condition != "loaded":
        raise ValueError("Object feedback requires real hand-object contacts")
    height_alignment = (
        SupportHeightAlignment(config["support_height_alignment"])
        if config.get("support_height_alignment")
        else None
    )
    if height_alignment and (feedback or condition != "loaded"):
        raise ValueError("Choose isolated support-height alignment on a loaded task")
    previous_command = targets[0].copy()
    tracking = (
        ArmTrackingCompensator(config["tracking_compensation"], speed_limits)
        if config.get("tracking_compensation")
        else None
    )
    if tracking and config.get("velocity_feedforward", False):
        raise ValueError(
            "Choose filtered tracking compensation or legacy velocity feedforward"
        )
    task = json.loads(Path(parent["configuration"]["task"]).read_text())
    reference_objects = []
    for record in parent["records"]:
        pose = np.eye(4)
        pose[:3, :3] = record["object_rotation"]
        pose[:3, 3] = record["object_position_m"]
        reference_objects.append(pose)
    reference_objects = np.array(reference_objects)
    feedback_rate_limited_steps = 0
    records, qstates, controls, timestamps = [], [], [], []
    velocities = []
    max_pen = max_arm_pen = max_error = 0.0
    max_speed_ratio, min_margin = 0.0, float("inf")
    saturation = np.zeros(len(names), int)
    peaks = np.zeros(len(names))
    steps = 0
    sample_index = 0
    aborted = None
    warning_names = [
        n for n in mujoco.mjtWarning.__members__ if n.startswith("mjWARN_")
    ]
    solver_warnings = dict.fromkeys(warning_names, 0)
    control_period, control_stride = controller_clock(
        model.opt.timestep, config.get("control_period_s")
    )
    command_lead_s = float(np.max(np.diff(times)))
    control_start = float(data.time)
    control_updates = 0
    closed = None
    if closed_spec is not None:
        from .closed_loop_grasp import ClosedLoopGrasp

        closed = ClosedLoopGrasp(
            model,
            data,
            roots,
            arms,
            qids,
            vids,
            ranges,
            speed_limits,
            parent,
            closed_spec["controller"],
            control_period,
            robot_geoms,
            arm_geoms,
            object_geoms,
            hands,
            dict(
                joint_margin_rad=audit["config"]["criteria"]["joint_margin_rad"],
                arm_penetration_m=audit["config"]["criteria"]["arm_penetration_m"],
                robot_penetration_m=parent["configuration"]["planning"][
                    "max_penetration_m"
                ],
            ),
        )
    if interaction is not None:
        if live_fingers is None or condition != "loaded":
            raise ValueError("Residual learning requires loaded live finger feedback")
        interaction.bind(
            model, data, live_fingers, roots, parent, times, control_period
        )
    while data.time < times[-1] - model.opt.timestep / 2:
        if steps % control_stride == 0:
            control_time = (
                control_start + control_updates * control_period
                if config.get("control_period_s") is not None
                else float(data.time)
            )
            control_updates += 1
            withdrawal = parent.get("continuous_protocol", {}).get("wrist_withdrawal")
            if withdrawal and motion_condition == "demonstrated":
                from .wrist_withdrawal import release_ready

                if live_fingers is None or live_fingers.release is None:
                    raise ValueError("Wrist withdrawal requires live supported release")
                if not release_ready(
                    control_time,
                    withdrawal,
                    parent["continuous_protocol"]["finger_release"],
                    live_fingers.release.started_s,
                    command_lead_s=command_lead_s,
                ):
                    aborted = "withdrawal_without_supported_unhooking"
                    break
            i = min(
                np.searchsorted(times, control_time, side="right") - 1, len(times) - 2
            )
            alpha = np.clip((control_time - times[i]) / (times[i + 1] - times[i]), 0, 1)
            pressure_offsets = None
            if interaction is not None:
                interaction.check_deadline()
                if (control_updates - 1) % interaction.policy_stride == 0:
                    action = yield interaction.observe(control_time, i, alpha)
                    interaction.accept(action)
                pressure_offsets = interaction.advance(
                    control_period, parent["records"][i]["phase"]
                )
            target = targets[i] * (1 - alpha) + targets[i + 1] * alpha
            if feedback or height_alignment:
                # Read the last solved contact forces before refreshing kinematics.
                normal = dict.fromkeys(roots, 0.0)
                opposed = {"thumb": 0.0, "fingers": 0.0}
                upward = 0.0
                for cid, contact in enumerate(data.contact):
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if not pair & object_geoms:
                        continue
                    for name, geoms in hands.items():
                        if pair & geoms:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, cid, force)
                            normal[name] += max(0, float(force[0]))
                            if name == "left":
                                upward += contact_wrench_on_body(model, data, cid, obj)[
                                    0
                                ][2]
                            else:
                                geom_name = (
                                    model.geom(next(iter(pair & geoms))).name or ""
                                )
                                if "thumb" in geom_name:
                                    opposed["thumb"] += max(0, float(force[0]))
                                elif any(
                                    g in geom_name
                                    for g in ("index", "middle", "ring", "pinky")
                                ):
                                    opposed["fingers"] += max(0, float(force[0]))
                mujoco.mj_kinematics(model, data)
                desired.qpos[:] = data.qpos
                desired.qpos[qids] = target
                mujoco.mj_kinematics(model, desired)
                mujoco.mj_comPos(model, desired)
                actual_object = object_transform(model, data)
                reference_object = reference_objects[i].copy()
                reference_object[:3, 3] = (1 - alpha) * reference_objects[
                    i, :3, 3
                ] + alpha * reference_objects[i + 1, :3, 3]
                delta = Rotation.from_matrix(
                    reference_objects[i + 1, :3, :3] @ reference_objects[i, :3, :3].T
                ).as_rotvec()
                reference_object[:3, :3] = (
                    Rotation.from_rotvec(alpha * delta).as_matrix()
                    @ reference_objects[i, :3, :3]
                )
                phase = parent["records"][i]["phase"]
                start = 0
                for arm in arms:
                    name = arm["name"]
                    body = roots[name]
                    width = len(arm["joints"])
                    indices = slice(start, start + width)
                    pose = np.eye(4)
                    pose[:3, :3] = desired.xmat[body].reshape(3, 3)
                    pose[:3, 3] = desired.xpos[body]
                    if height_alignment:
                        correction = np.zeros(6)
                        if name == "left":
                            correction[2] = height_alignment.update(
                                phase,
                                actual_object[:3, 3] - reference_object[:3, 3],
                                min(opposed.values()),
                                control_period,
                            )
                    else:
                        correction = feedback.update(
                            name,
                            actual_object,
                            reference_object,
                            pose,
                            control_period,
                            contact_n=min(opposed.values())
                            if name == "right"
                            else support_contact_evidence(
                                phase,
                                normal[name],
                                min(opposed.values()),
                                allow_precontact=config.get(
                                    "precontact_support_alignment", False
                                ),
                            ),
                            active=phase
                            in config.get("feedback_phases", {}).get(
                                name,
                                ("lift", "support_approach", "support_seat", "hold")
                                if name == "right"
                                else ("support_seat", "hold"),
                            ),
                            support=name == "left",
                            upward_force_n=upward,
                            weight_n=float(model.body_mass[obj] * 9.81),
                        )
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, desired, jp, jr, body)
                    target[indices] += joint_offset(
                        np.vstack([jp, jr])[:, vids[indices]], correction
                    )
                    start += width
                margin = audit["config"]["criteria"]["joint_margin_rad"]
                target = np.clip(target, ranges[:, 0] + margin, ranges[:, 1] - margin)
                delta = speed_limits * control_period
                bounded = np.clip(
                    target, previous_command - delta, previous_command + delta
                )
                feedback_rate_limited_steps += int(
                    not np.allclose(target, bounded, atol=1e-12, rtol=0)
                )
                target = bounded
            if closed is not None:
                target = closed.target(
                    control_time, control_period, target, previous_command, i, alpha
                )
            previous_command = target.copy()
            velocity = (
                (targets[i + 1] - targets[i]) / (times[i + 1] - times[i])
                if config.get("velocity_feedforward", False)
                else 0.0
            )
            if tracking:
                target, velocity = tracking.update(
                    target,
                    data.qpos[qids],
                    (targets[i + 1] - targets[i]) / (times[i + 1] - times[i]),
                    data.actuator_force[aids],
                    limits,
                    control_period,
                    active=parent["records"][i]["phase"]
                    in config.get("tracking_phases", ["hold"]),
                )
                margin = audit["config"]["criteria"]["joint_margin_rad"]
                target = np.clip(target, ranges[:, 0] + margin, ranges[:, 1] - margin)
            inertial_torque = 0.0
            if closed is not None and not closed.check_command(target):
                aborted = "closed_loop_collision_guard"
                break
            if inertia_control:
                mujoco.mj_fullM(model, mass_matrix, data.qM)
                acceleration = (1 - alpha) * accelerations[i] + alpha * accelerations[
                    i + 1
                ]
                inertial_torque = inertia_control.update(
                    mass_matrix[np.ix_(vids, vids)],
                    acceleration,
                    control_period,
                    active=parent["records"][i]["phase"]
                    in config.get("tracking_phases", ["hold"]),
                )
            data.ctrl[aids] = bias_compensated_target(
                target,
                data.qfrc_bias[vids] + inertial_torque,
                config["kp_nm_per_rad"],
                target_velocity=velocity,
                kd=config["kd_nm_s_per_rad"],
            )
            if live_fingers:
                release_observation = {}
                if (
                    live_fingers.release
                    and control_time >= live_fingers.release.settings["start_s"] - 0.5
                ):
                    actual_pose = object_transform(model, data)
                    object_geoms = set(np.where(model.geom_bodyid == obj)[0])
                    on_table = sum(
                        table in (c.geom1, c.geom2)
                        and bool(object_geoms & {c.geom1, c.geom2})
                        for c in data.contact[: data.ncon]
                    )
                    table_load = 0.0
                    for contact_index, contact in enumerate(data.contact[: data.ncon]):
                        if table in (contact.geom1, contact.geom2) and object_geoms & {
                            contact.geom1,
                            contact.geom2,
                        }:
                            contact_force = np.zeros(6)
                            mujoco.mj_contactForce(
                                model, data, contact_index, contact_force
                            )
                            table_load += max(0.0, float(contact_force[0]))
                    dof = model.joint("task_object_free").dofadr[0]
                    release_observation["table_observation"] = dict(
                        table_contacts=on_table,
                        clearance_m=float(
                            (
                                mesh.vertices @ actual_pose[:3, :3].T
                                + actual_pose[:3, 3]
                            )[:, 2].min()
                            - task["table"]["top_z_m"]
                        ),
                        speed_m_s=float(np.linalg.norm(data.qvel[dof : dof + 3])),
                        table_load_fraction=table_load
                        / float(model.body_mass[obj] * 9.81),
                    )
                if pressure_offsets is None:
                    live_fingers.command(
                        model, data, control_time, control_period, **release_observation
                    )
                else:
                    live_fingers.command(
                        model,
                        data,
                        control_time,
                        control_period,
                        pressure_offsets=pressure_offsets,
                        **release_observation,
                    )
            else:
                for src, dst in finger_pairs:
                    data.ctrl[dst] = (
                        saved["ctrl"][i, src] * (1 - alpha)
                        + saved["ctrl"][i + 1, src] * alpha
                    )
            if trim:
                forces = {"thumb": 0.0, trim.settings["contact_group"]: 0.0}
                for cid, contact in enumerate(data.contact):
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & object_geoms and pair & hands["right"]:
                        geom_name = (
                            model.geom(next(iter(pair & hands["right"]))).name or ""
                        )
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, cid, force)
                        for group in forces:
                            if group in geom_name:
                                forces[group] += max(0, float(force[0]))
                source, following = parent["records"][i : i + 2]
                active = (
                    source["phase"] == "hold"
                    and "reference_position_m" in source
                    and "reference_position_m" in following
                )
                error = 0.0
                if active:
                    goal = (1 - alpha) * np.asarray(
                        source["reference_position_m"]
                    ) + alpha * np.asarray(following["reference_position_m"])
                    error = float(np.linalg.norm(data.xpos[obj] - goal))
                data.ctrl[trim_ids] = trim.target(
                    data.ctrl[trim_ids],
                    force_n=forces[trim.settings["contact_group"]],
                    anchor_force_n=forces["thumb"],
                    reference_error_m=error,
                    active=active,
                    dt=control_period,
                )
        before_step_time = float(data.time)
        mujoco.mj_step(model, data)
        steps += 1
        solver_warnings = {
            n: int(data.warning[getattr(mujoco.mjtWarning, n)].number)
            for n in warning_names
        }
        health = step_rejection(
            before_step_time,
            float(data.time),
            float(model.opt.timestep),
            solver_warnings,
        )
        if health:
            aborted = health
            break
        saturation += abs(data.actuator_force[aids]) >= limits * 0.999
        peaks = np.maximum(peaks, abs(data.actuator_force[aids]))
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.max(abs(data.qvel)) > 100
        ):
            aborted = "unstable_dynamics"
            break
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError("External force assistance prohibited")
        max_error = max(max_error, float(np.max(abs(data.qpos[qids] - target))))
        max_speed_ratio = max(
            max_speed_ratio, float(np.max(abs(data.qvel[vids]) / speed_limits))
        )
        min_margin = min(
            min_margin,
            float(
                np.minimum(
                    data.qpos[qids] - ranges[:, 0], ranges[:, 1] - data.qpos[qids]
                ).min()
            ),
        )
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & robot_geoms:
                bodies = [
                    int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
                ]
                if pair & arm_geoms and joint_neighbors(model, *bodies):
                    continue
                max_pen = max(max_pen, -float(contact.dist))
                if pair & arm_geoms:
                    max_arm_pen = max(max_arm_pen, -float(contact.dist))
        if data.time + model.opt.timestep / 2 >= times[sample_index]:
            # Collect solved contact forces before refreshing kinematics.
            groups = dict.fromkeys(
                ("thumb", "index", "middle", "ring", "pinky", "palm_or_other"), 0.0
            )
            support_force = np.zeros(3)
            table_contacts = 0
            for cid, contact in enumerate(data.contact):
                pair = {int(contact.geom1), int(contact.geom2)}
                if not pair & object_geoms:
                    continue
                table_contacts += int(table in pair)
                if pair & hands["left"]:
                    force, _ = contact_wrench_on_body(model, data, cid, obj)
                    support_force += force
                if pair & hands["right"]:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, cid, force)
                    geom = next(iter(pair & hands["right"]))
                    name = model.geom(geom).name or ""
                    group = next(
                        (g for g in groups if g != "palm_or_other" and g in name),
                        "palm_or_other",
                    )
                    groups[group] += max(0.0, float(force[0]))
            mujoco.mj_kinematics(model, data)
            pose = object_transform(model, data)
            source = parent["records"][sample_index]
            record = dict(
                time_s=float(data.time),
                phase=source["phase"],
                minimum_object_height_m=float(
                    (mesh.vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()
                ),
                object_position_m=pose[:3, 3].tolist(),
                object_rotation=pose[:3, :3].tolist(),
                rotation_change_deg=float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            pose[:3, :3] @ initial_rotation.T
                        ).magnitude()
                    )
                ),
                wrist_position_m=data.xpos[roots["right"]].tolist(),
                wrist_rotation=data.xmat[roots["right"]].reshape(3, 3).tolist(),
                support_wrist_position_m=data.xpos[roots["left"]].tolist(),
                support_wrist_rotation=data.xmat[roots["left"]].reshape(3, 3).tolist(),
                thumb_force_n=groups["thumb"],
                finger_force_n=sum(
                    groups[g] for g in ("index", "middle", "ring", "pinky")
                ),
                normal_contact_force_by_group_n=groups,
                support_object_force_world_n=support_force.tolist(),
                table_contacts=table_contacts,
            )
            # Measure Cartesian tracking against frozen audited arm commands;
            # no desired state is ever written into the simulated robot.
            desired.qpos[:] = data.qpos
            desired.qpos[qids] = targets[sample_index]
            mujoco.mj_kinematics(model, desired)
            record["wrist_tracking"] = wrist_errors(data, desired, roots)
            if parent.get("continuous_protocol", {}).get("wrist_withdrawal") and source[
                "phase"
            ] in ("release", "released_settle"):
                from .arm_finger_plan import hand_table_clearance

                record["hand_table_clearance_m"] = hand_table_clearance(
                    model, data, hands["left"] | hands["right"], table
                )
            desired.qpos[qids] = target
            mujoco.mj_kinematics(model, desired)
            record["commanded_wrist_tracking"] = wrist_errors(data, desired, roots)
            record["joint_tracking_error_rad"] = (
                data.qpos[qids] - targets[sample_index]
            ).tolist()
            record["arm_torque_nm"] = data.actuator_force[aids].tolist()
            if tracking:
                record["tracking_compensation"] = tracking.diagnostics()
            if trim:
                record["finger_contact_trim"] = trim.diagnostics()
            if live_fingers:
                record.update(live_fingers.diagnostics())
            if contact_motion:
                # Re-solve on a separate data object at the saved physical state.
                # Neither refreshed contacts nor solver warmstarts feed the run.
                diagnostic_data.time = data.time
                for field in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
                    getattr(diagnostic_data, field)[:] = getattr(data, field)
                mujoco.mj_forward(model, diagnostic_data)
                record["contact_patch_motion"] = contact_motion.sample(
                    model, diagnostic_data
                )
                velocities.append(data.qvel.copy())
            if height_alignment:
                record["support_height_alignment"] = height_alignment.diagnostics()
            if feedback:
                record["object_feedback"] = feedback.diagnostics()
            if "reference_position_m" in source:
                record["reference_position_error_m"] = float(
                    np.linalg.norm(pose[:3, 3] - source["reference_position_m"])
                )
                record["reference_rotation_error_deg"] = float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            pose[:3, :3] @ np.asarray(source["reference_rotation"]).T
                        ).magnitude()
                    )
                )
            records.append(record)
            if interaction is not None:
                interaction.record(record)
            if closed is not None:
                closed.record(record)
            if descent is not None:
                descent.record(record)
            qstates.append(data.qpos.copy())
            controls.append(data.ctrl.copy())
            timestamps.append(data.time)
            sample_index = min(sample_index + 1, len(times) - 1)
            if record["minimum_object_height_m"] < -0.04:
                aborted = "object_dropped"
                break
    metrics = measure_hold(records, table_top_m=task["table"]["top_z_m"])
    metrics.update(support_hold_metrics(records, float(model.body_mass[obj] * 9.81)))
    continuous_metrics, continuous_failures = None, []
    if parent.get("continuous_protocol"):
        from .continuous_reference import evaluate_continuous

        continuous_metrics, continuous_failures = evaluate_continuous(
            records,
            parent["continuous_protocol"],
            task["table"]["top_z_m"],
            stationary=motion_condition == "stationary_hold",
        )
        for key in ("hold_slip_m", "support_wrist_drift_m"):
            if key in continuous_metrics:
                metrics[key] = continuous_metrics[key]
    failures = hold_rejections(
        metrics,
        minimum_clearance_m=task["goal"]["minimum_clearance_m"],
        observed_max_penetration_m=max_pen,
        allowed_max_penetration_m=parent["configuration"]["planning"][
            "max_penetration_m"
        ],
        aborted=aborted,
    )
    failures.extend(support_rejections(metrics))
    failures.extend(continuous_failures)
    if stationary_admission and stationary_admission.get("diagnostic_only"):
        failures.append("diagnostic_motion_without_stationary_promotion")
    if condition != "loaded":
        failures.append("diagnostic_contact_condition")
    reference = [
        r
        for r in records
        if (r["phase"] == "hold" or parent.get("continuous_protocol"))
        and "reference_position_error_m" in r
    ]
    if (
        not reference
        or max(r["reference_position_error_m"] for r in reference) > 0.02
        or max(r["reference_rotation_error_deg"] for r in reference) > 10
    ):
        failures.append("reference_fidelity")
    if max_arm_pen > audit["config"]["criteria"]["arm_penetration_m"]:
        failures.append("arm_penetration")
    if max_speed_ratio > 1:
        failures.append("arm_velocity_limit")
    if min_margin < audit["config"]["criteria"]["joint_margin_rad"]:
        failures.append("arm_joint_margin")
    if aborted or data.time < times[-1] - model.opt.timestep:
        failures.append("incomplete_pilot")
    state_arrays = dict(qpos=qstates, ctrl=controls, timestamps_s=timestamps)
    if contact_motion:
        state_arrays["qvel"] = velocities
    np.savez_compressed(out / "states.npz", **state_arrays)
    motion_summary = None
    if contact_motion:
        from .contact_patch_motion import summarize

        motion_summary = summarize(records, config["contact_patch_motion"])
    report = dict(
        schema="real2sim2real.arm-replay-pilot/1",
        motion_condition=motion_condition,
        demonstrated_motion_executed=motion_condition == "demonstrated",
        finger_plan_admission=finger_plan_admission,
        finger_path_screen=finger_path,
        acquisition_command_path=acquisition_path,
        stationary_admission=stationary_admission,
        finger_control_mode=finger_mode,
        controller_period_s=control_period,
        controller_updates=control_updates,
        physics_timestep_s=float(model.opt.timestep),
        fixed_controller_clock=config.get("control_period_s") is not None,
        live_finger_source_hashes=live_fingers.source_hashes if live_fingers else None,
        live_finger_feedback_final=live_fingers.diagnostics() if live_fingers else None,
        live_finger_feedback_settings=live_fingers.settings if live_fingers else None,
        contact_patch_motion_summary=motion_summary,
        contact_patch_motion_observations=(
            "Separate-data forward solve at saved qpos/qvel/ctrl and warmstart; "
            "sampled diagnostic, not per-step contact history"
            if contact_motion
            else None
        ),
        config=config,
        records=records,
        **metrics,
        success=not failures,
        rejected_for=failures,
        physics_run=True,
        hardware_ready=False,
        max_robot_penetration_m=max_pen,
        max_arm_penetration_m=max_arm_pen,
        maximum_joint_tracking_error_rad=max_error,
        maximum_dynamic_velocity_ratio=max_speed_ratio,
        minimum_dynamic_joint_margin_rad=min_margin
        if np.isfinite(min_margin)
        else None,
        solver_warning_counts=solver_warnings,
        simulation_health_implementation_sha256=digest(
            snapshot / "simulation_health.py"
        ),
        contact_exclusions=contact_exclusions,
        wrist_tracking_by_phase=phase_maxima(records, "wrist_tracking"),
        commanded_wrist_tracking_by_phase=phase_maxima(
            records, "commanded_wrist_tracking"
        ),
        torque_peak_nm=dict(zip(names, peaks.tolist())),
        saturation_fraction=dict(zip(names, (saturation / max(steps, 1)).tolist())),
        control_mode="bounded joint PD plus bias compensation; "
        + (
            "live finger pressure feedback; "
            if live_fingers
            else "archived fingers with bounded contact trim; "
            if trim
            else "archived finger commands; "
        )
        + (
            "bounded object-relative feedback"
            if feedback
            else "latched precontact support-height alignment"
            if height_alignment
            else "bounded fixed-anchor slip and coordinated object-tilt feedback"
            if closed is not None
            else "no online object feedback"
        ),
        object_feedback=None if feedback is None else feedback.diagnostics(),
        feedback_command_rate_limited_steps=feedback_rate_limited_steps,
        tracking_compensation=None if tracking is None else tracking.diagnostics(),
        inertial_feedforward_peak_nm=None
        if inertia_control is None
        else inertia_control.peak.tolist(),
        feedback_observation_source="privileged simulator poses and solved contact forces"
        if feedback or height_alignment or trim or closed is not None
        else None,
        fixture_joints_removed=True,
        object_state_written_only_at_initialization=True,
        sources={
            str(audit_path): digest(audit_path),
            str(source_scene): digest(source_scene),
            str(archive): digest(archive),
        },
        scene_sha256=digest(scene),
        states_sha256=digest(out / "states.npz"),
        implementation_sha256=digest(snapshot / "arm_replay_pilot.py"),
        tracking_implementation_sha256=digest(snapshot / "arm_tracking_control.py"),
        robustness_implementation_sha256=digest(snapshot / "arm_robustness.py"),
        simulation_case=simulation_case,
        finger_contact_trim=None if trim is None else trim.diagnostics(),
        precontact_support_alignment=config.get("precontact_support_alignment", False),
        support_height_alignment=None
        if height_alignment is None
        else height_alignment.diagnostics(),
        support_height_alignment_observations="privileged simulator pose and primary contact at insertion start"
        if height_alignment
        else None,
        finger_contact_trim_observations="privileged simulator normal forces and object pose"
        if trim
        else None,
        finger_contact_trim_implementation_sha256=digest(
            snapshot / "finger_contact_trim.py"
        ),
        diagnostics_implementation_sha256=digest(
            snapshot / "arm_replay_diagnostics.py"
        ),
        feedback_implementation_sha256=digest(snapshot / "arm_object_feedback.py"),
    )
    if interaction is not None:
        report["residual_adapter"] = interaction.finish(report, out)
    if closed is not None:
        report["closed_loop_grasp_feedback"] = closed.finish(out)
        report["closed_loop_grasp_feedback"]["implementation_sha256"] = {
            name: digest(snapshot / name)
            for name in ("closed_loop_grasp.py", "slip_tilt_feedback.py")
        }
    if descent is not None:
        report["descent_pad_control"] = descent.finish(out)
        report["descent_pad_control"]["implementation_sha256"] = digest(
            snapshot / "descent_pad_control.py"
        )
    if correction_spec is not None:
        report["bounded_grasp_correction"] = dict(
            spec=correction_spec,
            path=correction_path,
            promotion_eligible=False,
            correction_is_learned=False,
            finger_owner="existing live pressure servo nominal target",
            release_owner="original table-supported release; command offsets recover to zero before release",
            command_targets_sha256=digest(out / "bounded-command-targets.npz"),
            implementation_sha256=digest(snapshot / "bounded_grasp_correction.py"),
        )
    if parent.get("continuous_protocol"):
        report["continuous_protocol"] = parent["continuous_protocol"]
        report["continuous_metrics"] = continuous_metrics
    write_json(out / "result.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in (
                    "success",
                    "rejected_for",
                    "maximum_joint_tracking_error_rad",
                    "hold_minimum_clearance_m",
                    "hold_slip_m",
                )
            }
        ),
        flush=True,
    )
    return report


def run(config):
    """Run the unchanged noninteractive controller to completion."""
    worker = rollout(config)
    try:
        next(worker)
    except StopIteration as done:
        return done.value
    finally:
        worker.close()
    raise RuntimeError("A frozen replay cannot request a policy action")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if not a.worker:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(a.config.resolve()),
                "--worker",
            ],
            timeout=min(config["deadline_s"], 900),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    run(config)


if __name__ == "__main__":
    main()
