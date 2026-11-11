"""Matched free-object fixture holds from pinned paired grasp hypotheses.

The staged diagnostic and continuous-prefix diagnostic have distinct clocks.
Neither promotes a controller or substitutes for an arm-mounted hold.
"""

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json


def clock_counts(step, feedback, sample, duration):
    values = np.asarray([step, feedback, sample, duration], float)
    if (
        not np.isfinite(values).all()
        or not 0.0001 <= step <= 0.001
        or not step <= feedback <= 0.01
        or not feedback <= sample <= 0.025
        or not 5 <= duration <= 30
    ):
        raise ValueError("Bounded finite fixture clocks required")
    ratios = np.array([feedback, sample]) / step
    if not np.allclose(ratios, np.rint(ratios), atol=1e-9, rtol=0):
        raise ValueError("Feedback and sampling must be integer physics multiples")
    return int(round(ratios[0])), int(round(ratios[1])), int(np.ceil(duration / step))


class Commands:
    """Own robot commands only; reference object coordinates are never actuated."""

    def __init__(
        self, model, source, hands, settled, condition, acquisition_blend_s=0.0
    ):
        import mujoco
        from real2sim2real.retargeting.grasp_kinematics import robot_indices

        self.model, self.scratch = model, mujoco.MjData(model)
        self.hands, self.condition = hands, condition
        if not np.isfinite(acquisition_blend_s) or not 0 <= acquisition_blend_s <= 0.5:
            raise ValueError("Acquisition command blend must be within half a second")
        self.blend = acquisition_blend_s
        self.indices = [robot_indices(model, h["root_body"]) for h in hands]
        self.rollout = {
            k: v
            for k, v in source["configuration"]["rollout"].items()
            if k != "finger_release"
        }
        self.anchor = np.asarray(settled["settled_world_from_object"])
        self.inverse_anchor = np.linalg.inv(self.anchor)
        self.protocol = deepcopy(source["continuous_protocol"])
        rows = source["records"]
        self.times = np.array([r["time_s"] for r in rows])
        self.positions = np.array([r["object_position_m"] for r in rows])
        self.rotation = Slerp(
            self.times, Rotation.from_matrix([r["object_rotation"] for r in rows])
        )
        self.end_close = self.rollout["approach_s"] + self.rollout["close_s"]
        if condition == "continuous_prefix_hold":
            self.hold_start = next(r["time_s"] for r in rows if r["phase"] == "hold")
            self.duration = self.protocol["duration_s"]
        elif condition == "staged_hold":
            self.hold_start = self.end_close + 2.0
            self.duration = self.hold_start + self.rollout["hold_s"]
        else:
            raise ValueError("Use continuous_prefix_hold or staged_hold")
        self.scratch.qpos[
            model.joint("task_object_free").qposadr[0] : model.joint(
                "task_object_free"
            ).qposadr[0]
            + 7
        ] = settled["object_qpos"]

    def goal(self, time):
        from .object_grasp_pilot import smoothstep

        goal = self.anchor.copy()
        if self.condition == "staged_hold":
            goal[2, 3] += 0.07 * smoothstep((time - self.end_close) / 2)
        else:
            t = np.clip(time, self.times[0], self.hold_start)
            goal[:3, 3] = [
                np.interp(t, self.times, self.positions[:, i]) for i in range(3)
            ]
            goal[:3, :3] = self.rotation(t).as_matrix()
        if self.condition == "continuous_prefix_hold" and self.blend:
            a = smoothstep((time - self.end_close) / self.blend)
            goal[:3, 3] = self.anchor[:3, 3] + a * (goal[:3, 3] - self.anchor[:3, 3])
            delta = Rotation.from_matrix(goal[:3, :3] @ self.anchor[:3, :3].T)
            goal[:3, :3] = (
                Rotation.from_rotvec(a * delta.as_rotvec()).as_matrix()
                @ self.anchor[:3, :3]
            )
        return goal

    def at(self, time):
        import mujoco
        from .arm_finger_feedback import finger_waypoint

        data, model = self.scratch, self.model
        change = self.goal(time) @ self.inverse_anchor
        for hand, idx in zip(self.hands, self.indices):
            data.qpos[idx["q"]] = finger_waypoint(
                time,
                *(hand[k] for k in ("q_pregrasp", "q_open", "q_closed")),
                self.rollout,
            )
            mujoco.mj_kinematics(model, data)
            root = idx["root"]
            wrist = np.eye(4)
            wrist[:3, 3] = data.xpos[root]
            wrist[:3, :3] = data.xmat[root].reshape(3, 3)
            target = change @ wrist
            qids = [
                model.joint(hand["fixture_prefix"] + s).qposadr[0]
                for s in ("x", "y", "z", "rx", "ry", "rz")
            ]
            data.qpos[qids[:3]] += target[:3, 3] - wrist[:3, 3]
            base = Rotation.from_quat(model.body_quat[root][[1, 2, 3, 0]]).as_matrix()
            data.qpos[qids[3:]] = Rotation.from_matrix(
                base.T @ target[:3, :3]
            ).as_euler("XYZ")
        return data.qpos.copy()


def run(config):
    import mujoco
    import trimesh
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_root_frame,
        command,
        object_transform,
        verify_unassisted_model,
    )
    from .pad_grasp_planning import path_audit
    from .contact_force_control import BilateralContactForce, object_pad_observations
    from .support_control import (
        contact_wrench_on_body,
        bias_compensated_target,
        support_hold_metrics,
        support_rejections,
    )
    from .grasp_hold import measure_hold, hold_rejections
    from .continuous_reference import evaluate_continuous
    from real2sim2real.retargeting.grasp_kinematics import contact_penetrations

    acquisition = pinned_json(config["acquisition_audit"])
    if not acquisition["geometric_path_pass"]:
        raise ValueError("Paired geometric preflight required")
    plan = pinned_json(acquisition["config"]["plan"])
    source = pinned_json(plan["config"]["source_trial"])
    source_path = Path(plan["config"]["source_trial"]["path"])
    if digest(source_path.with_name("states.npz")) != source["states_sha256"]:
        raise ValueError("Source command states changed")
    base = source["configuration"]
    candidate = deepcopy(source["candidate"])
    if config["grasp"] == "replacement":
        candidate["support"] = deepcopy(
            plan["candidates"][acquisition["config"]["candidate_index"]]
        )
    elif config["grasp"] != "retained":
        raise ValueError("Explicit retained/replacement grasp required")
    candidate.update(root_body=base["robot"]["root_body"], fixture_prefix="fixture_")
    hands = [candidate, candidate["support"]]
    model, data, mesh = load_scene(base)
    verify_unassisted_model(model)
    model.opt.timestep = config["physics_step_s"]
    settled = load_plan(base)["settling"]
    for h in hands:
        set_root_frame(
            model,
            data,
            h["root_body"],
            h["fixture_prefix"],
            h["base_position_m"],
            h["base_rotation"],
            preserve_state=True,
        )
    commands = Commands(
        model,
        source,
        hands,
        settled,
        config["condition"],
        config.get("acquisition_blend_s", 0.0),
    )
    update_every, sample_every, steps = clock_counts(
        config["physics_step_s"], config["feedback_period_s"], 0.02, commands.duration
    )
    indices = commands.indices
    combined = {
        k: np.concatenate([idx[k] for idx in indices])
        for k in ("q", "joints", "dofs", "geoms")
    }
    for h, idx in zip(hands, indices):
        if h["joint_names"] != [model.joint(int(j)).name for j in idx["joints"]]:
            raise ValueError("Native joint order changed")
    objq = model.joint("task_object_free").qposadr[0]
    data.qpos[:] = commands.at(0)
    data.qpos[objq : objq + 7] = settled["object_qpos"]
    mujoco.mj_forward(model, data)
    times = np.r_[np.arange(0, commands.end_close, 0.02), commands.end_close]
    qs = np.array([commands.at(t)[combined["q"]] for t in times])
    path = path_audit(model, data, combined, qs)
    full_times = np.r_[np.arange(0, commands.hold_start, 0.02), commands.hold_start]
    full_q = np.array([commands.at(t)[combined["q"]] for t in full_times])
    bounds = model.jnt_range[combined["joints"]]
    path["lift_command_within_joint_limits"] = bool(
        ((full_q >= bounds[:, 0]) & (full_q <= bounds[:, 1])).all()
    )
    path["collision_scope"] = (
        "Actual commanded acquisition against initially settled object; moving object/contact response tested by dynamics"
    )

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    helpers = [
        Path(__file__),
        Path(inspect.getfile(load_scene)),
        Path(inspect.getfile(BilateralContactForce)),
        Path(inspect.getfile(Commands.at)),
        Path(inspect.getfile(evaluate_continuous)),
        Path(inspect.getfile(path_audit)),
        Path(inspect.getfile(bias_compensated_target)),
    ]
    snapshots = out / "implementation"
    snapshots.mkdir()
    hashes = {str(p): digest(p) for p in helpers}
    for p in set(helpers):
        shutil.copyfile(p, snapshots / p.name)
    protocol = dict(
        config=config,
        acquisition_end_s=commands.end_close,
        hold_start_s=commands.hold_start,
        duration_s=commands.duration,
        staged_lift_m=0.07 if config["condition"] == "staged_hold" else None,
        reference_preserved=config["condition"] == "continuous_prefix_hold",
        acquisition_command_blend_s=commands.blend,
        reference_clock_changed=False,
        wrist_controller="Position-controlled fixture targets; no arm dynamics or old clearance/lift feedback",
        release_tested=False,
        physical_geometry_verified=False,
        object_state_written_only_at_initialization=True,
        fixture_actuation="Prepared force-limited Cartesian wrists; unchanged native finger actuators",
        implementation_sources=hashes,
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "command-path.json", path)
    if (
        not path["within_joint_limits"]
        or not path["lift_command_within_joint_limits"]
        or path["max_penetration_m"] > base["planning"]["max_penetration_m"]
    ):
        result = dict(
            schema="fixture-hold/1",
            config=config,
            success=False,
            physics_run=False,
            rejected_for=["command_path_rejected"],
            command_path=path,
        )
        write_json(out / "result.json", result)
        return result
    world_mesh = mesh.copy()
    world_mesh.apply_transform(commands.anchor)
    servo = BilateralContactForce(
        model,
        data,
        base["contact_force_feedback"],
        candidate,
        commands.anchor,
        world_mesh,
    )
    obj = model.body("task_object").id
    task = json.loads(Path(base["task"]).read_text())
    table_z = task["table"]["top_z_m"]
    activation = (
        commands.rollout["approach_s"]
        + base["contact_force_feedback"]["close_activation_fraction"]
        * commands.rollout["close_s"]
    )
    owned = set(map(int, combined["geoms"]))
    secondary = set(map(int, indices[1]["geoms"]))
    pad_ids = [p["gid"] for p in servo.pads]
    records = []
    states = []
    velocities = []
    controls = []
    max_pen = 0.0
    aborted = None
    for step in range(steps + 1):
        time = step * model.opt.timestep
        if step < steps and step % update_every == 0:
            target = servo.target(
                model,
                data,
                commands.at(time),
                active=time >= activation,
                dt=config["feedback_period_s"],
                retention_active=time >= commands.end_close,
            )
            if base["support_controller"].get("actuator_bias_compensation"):
                target = bias_compensated_target(model, data, target, indices[1])
            command(model, data, target)
        max_pen = max(
            max_pen, float(contact_penetrations(model, data, combined).max(initial=0))
        )
        if step % sample_every == 0 or step == steps:
            mujoco.mj_forward(model, data)
            pose = object_transform(model, data)
            height = float(trimesh.transform_points(mesh.vertices, pose)[:, 2].min())
            obs = object_pad_observations(model, data, obj, pad_ids)
            pad_groups = [
                {
                    p["group"]: obs[p["gid"]]["force_n"]
                    for p in servo.pads
                    if p["role"] == role
                }
                for role in ("primary", "secondary")
            ]
            groups = [
                dict.fromkeys(
                    ("thumb", "index", "middle", "ring", "pinky", "palm_or_other"), 0.0
                )
                for _ in range(2)
            ]
            support_force = np.zeros(3)
            tables = 0
            for c, contact in enumerate(data.contact):
                pair = {int(contact.geom1), int(contact.geom2)}
                if not any(model.geom_bodyid[g] == obj for g in pair):
                    continue
                if pair & secondary:
                    support_force += contact_wrench_on_body(model, data, c, obj)[0]
                for role, idx in enumerate(indices):
                    owned_contact = pair & set(map(int, idx["geoms"]))
                    if owned_contact:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, c, force)
                        name = model.geom(next(iter(owned_contact))).name or ""
                        group = next(
                            (
                                g
                                for g in groups[role]
                                if g != "palm_or_other" and g in name
                            ),
                            "palm_or_other",
                        )
                        groups[role][group] += max(0.0, float(force[0]))
                if not pair & owned:
                    tables += 1
            row = dict(
                time_s=float(data.time),
                phase="hold"
                if data.time >= commands.hold_start
                else "approach"
                if data.time < commands.rollout["approach_s"]
                else "close"
                if data.time < commands.end_close
                else "lift",
                minimum_object_height_m=height,
                object_position_m=pose[:3, 3].tolist(),
                object_rotation=pose[:3, :3].tolist(),
                rotation_change_deg=float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            pose[:3, :3] @ commands.anchor[:3, :3].T
                        ).magnitude()
                    )
                ),
                thumb_force_n=groups[0]["thumb"],
                finger_force_n=sum(
                    groups[0][f] for f in ("index", "middle", "ring", "pinky")
                ),
                normal_contact_force_by_group_n=groups[0],
                support_normal_contact_force_by_group_n=groups[1],
                support_object_force_world_n=support_force.tolist(),
                table_contacts=tables,
                pad_normal_forces_n=pad_groups,
                contact_force_feedback=servo.diagnostics(),
            )
            for prefix, idx in zip(("", "support_"), indices):
                row[prefix + "wrist_position_m"] = data.xpos[idx["root"]].tolist()
                row[prefix + "wrist_rotation"] = (
                    data.xmat[idx["root"]].reshape(3, 3).tolist()
                )
            records.append(row)
            states.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            controls.append(data.ctrl.copy())
            if height < table_z - 0.04 or not np.isfinite(data.qpos).all():
                aborted = "dropped_or_unstable"
                break
        if step < steps:
            mujoco.mj_step(model, data)
        if step % (sample_every * 100) == 0:
            print(
                json.dumps(dict(time_s=float(data.time), duration_s=commands.duration)),
                flush=True,
            )
    metrics = measure_hold(records, table_top_m=table_z)
    weight = float(model.body_mass[obj] * -model.opt.gravity[2])
    metrics.update(support_hold_metrics(records, weight))
    anchored, failures = evaluate_continuous(
        records, commands.protocol, table_z, stationary=True
    )
    hold_only = {k: metrics[k] for k in ("hold_slip_m", "support_wrist_drift_m")}
    metrics.update(anchored)
    metrics["hold_only_drift_diagnostic"] = hold_only
    failures += hold_rejections(
        metrics,
        minimum_clearance_m=task["goal"]["minimum_clearance_m"],
        observed_max_penetration_m=max_pen,
        allowed_max_penetration_m=base["planning"]["max_penetration_m"],
        aborted=aborted,
    )
    failures += support_rejections(metrics)
    load = metrics["support_load_fraction"]
    if data.time < commands.duration - 1e-6:
        failures.append("incomplete_duration")
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    if warnings:
        failures.append("simulation_warning")
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError("Hold implementation changed")
    pinned_json(config["acquisition_audit"])
    pinned_json(acquisition["config"]["plan"])
    pinned_json(plan["config"]["source_trial"])
    np.savez_compressed(
        out / "states.npz",
        qpos=states,
        qvel=velocities,
        ctrl=controls,
        timestamps_s=[r["time_s"] for r in records],
    )
    mujoco.mj_saveLastXML(str(out / "scene.xml"), model)
    result = dict(
        schema="fixture-hold/1",
        config=config,
        configuration=base,
        candidate=candidate,
        success=not failures,
        physics_run=True,
        rejected_for=sorted(set(failures)),
        records=records,
        states_sha256=digest(out / "states.npz"),
        scene_sha256=digest(out / "scene.xml"),
        metrics=metrics,
        max_robot_penetration_m=max_pen,
        support_load_fraction=load,
        simulation_final_time_s=float(data.time),
        simulation_requested_duration_s=commands.duration,
        solver_warnings=warnings,
        protocol=protocol,
        command_path=path,
        control_admitted=False,
        arm_reachability_validated=False,
    )
    write_json(out / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "success",
                    "rejected_for",
                    "metrics",
                    "support_load_fraction",
                    "max_robot_penetration_m",
                )
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Bounded CPU deadline required")
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded

        bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=config["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    main()
