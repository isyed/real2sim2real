"""Bounded two-arm joint control of a screened saved grasp trajectory on CPU.

This is an initial transfer diagnostic. Finger commands replay the source trial;
arm motors use bounded PD plus model bias compensation. Source fixture feedback
is not reconstructed as a new object-feedback controller. The object is free.
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


def run(config):
    import mujoco

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
    arm_bodies = {model.body(n).id for a in arms for n in a["collision_bodies"]}
    arm_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies
    }
    robot_geoms = arm_geoms | hands["left"] | hands["right"]
    records, qstates, controls, timestamps = [], [], [], []
    max_pen = max_arm_pen = max_error = 0.0
    max_speed_ratio, min_margin = 0.0, float("inf")
    saturation = np.zeros(len(names), int)
    peaks = np.zeros(len(names))
    steps = 0
    sample_index = 0
    aborted = None
    while data.time < times[-1] - model.opt.timestep / 2:
        i = min(np.searchsorted(times, data.time, side="right") - 1, len(times) - 2)
        alpha = np.clip((data.time - times[i]) / (times[i + 1] - times[i]), 0, 1)
        target = targets[i] * (1 - alpha) + targets[i + 1] * alpha
        velocity = (
            (targets[i + 1] - targets[i]) / (times[i + 1] - times[i])
            if config.get("velocity_feedforward", False)
            else 0.0
        )
        data.ctrl[aids] = bias_compensated_target(
            target,
            data.qfrc_bias[vids],
            config["kp_nm_per_rad"],
            target_velocity=velocity,
            kd=config["kd_nm_s_per_rad"],
        )
        for src, dst in finger_pairs:
            data.ctrl[dst] = (
                saved["ctrl"][i, src] * (1 - alpha) + saved["ctrl"][i + 1, src] * alpha
            )
        mujoco.mj_step(model, data)
        steps += 1
        saturation += abs(data.actuator_force[aids]) >= limits * 0.999
        peaks = np.maximum(peaks, abs(data.actuator_force[aids]))
        if not np.isfinite(data.qpos).all() or np.max(abs(data.qvel)) > 100:
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
            qstates.append(data.qpos.copy())
            controls.append(data.ctrl.copy())
            timestamps.append(data.time)
            sample_index = min(sample_index + 1, len(times) - 1)
            if record["minimum_object_height_m"] < -0.04:
                aborted = "object_dropped"
                break
    task = json.loads(Path(parent["configuration"]["task"]).read_text())
    metrics = measure_hold(records, table_top_m=task["table"]["top_z_m"])
    metrics.update(support_hold_metrics(records, float(model.body_mass[obj] * 9.81)))
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
    reference = [
        r for r in records if r["phase"] == "hold" and "reference_position_error_m" in r
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
    np.savez_compressed(
        out / "states.npz", qpos=qstates, ctrl=controls, timestamps_s=timestamps
    )
    report = dict(
        schema="real2sim2real.arm-replay-pilot/1",
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
        minimum_dynamic_joint_margin_rad=min_margin,
        torque_peak_nm=dict(zip(names, peaks.tolist())),
        saturation_fraction=dict(zip(names, (saturation / max(steps, 1)).tolist())),
        control_mode="bounded joint PD plus bias compensation; archived finger commands; no online object feedback",
        fixture_joints_removed=True,
        object_state_written_only_at_initialization=True,
        sources={
            str(audit_path): digest(audit_path),
            str(source_scene): digest(source_scene),
            str(archive): digest(archive),
        },
        scene_sha256=digest(scene),
        states_sha256=digest(out / "states.npz"),
        implementation_sha256=digest(Path(__file__)),
    )
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
