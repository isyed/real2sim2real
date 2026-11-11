"""Joint wrist/finger closure proposals from a pinned loaded arm observation.

The saved nominal-minus-actual finger displacement is an empirical compliance
approximation, not a force model. Optimization is position-only. Admission still
requires new full-arm paths and complete free-object stationary dynamics.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json


def command_bounds(nominal, deflection, limits, maximum_change):
    """Keep both proposed command and predicted loaded posture within native limits."""
    nominal, deflection, limits = map(np.asarray, (nominal, deflection, limits))
    lo = np.maximum.reduce(
        [limits[:, 0], nominal - maximum_change, limits[:, 0] + deflection]
    )
    hi = np.minimum.reduce(
        [limits[:, 1], nominal + maximum_change, limits[:, 1] + deflection]
    )
    if (
        not np.isfinite(
            np.r_[nominal, deflection, limits.ravel(), maximum_change]
        ).all()
        or not 0 < maximum_change <= 0.35
        or np.any(lo >= hi)
    ):
        raise ValueError(
            "Nonempty bounded command and loaded-posture intersection required"
        )
    return lo, hi


def offsets(before, after, object_rotation):
    return dict(
        translation=object_rotation.T @ (after[:3, 3] - before[:3, 3]),
        rotation=object_rotation.T
        @ Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec(),
    )


def response_deflection(command, actual, bias_shift, fraction):
    command, actual, bias_shift = [
        np.asarray(v, float) for v in (command, actual, bias_shift)
    ]
    if (
        command.ndim != 1
        or actual.shape != command.shape
        or bias_shift.shape != command.shape
        or not np.isfinite(np.r_[command, actual, bias_shift, fraction]).all()
        or not 0 <= fraction <= 1
    ):
        raise ValueError(
            "Finite matching loaded motor observations and bounded response fraction required"
        )
    return fraction * (command - bias_shift - actual)


def environment_excess(model, data, robot, arms, objects):
    """Same environment/self limits for commanded and predicted loaded poses."""
    from .arm_path_audit import joint_neighbors

    order = sorted(robot)
    lookup = {g: j for j, g in enumerate(order)}
    result = np.zeros(len(order))
    for c in data.contact:
        pair = {int(c.geom1), int(c.geom2)}
        if not pair & robot or pair & objects:
            continue
        if pair & arms and joint_neighbors(
            model, int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        ):
            continue
        limit = 0.001 if pair & arms else 0.003
        for g in pair & robot:
            result[lookup[g]] = max(result[lookup[g]], -float(c.dist) - limit)
    return result


def run(config):
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from .arm_finger_feedback import ArmFingerFeedback
    from .arm_replay_pilot import subtree_geoms
    from .object_grasp_pilot import load_scene
    from .pad_seating_plan import nonpad_chain_geoms, contact_witness
    from .closing_wrist_offsets import validate
    from .arm_finger_plan import hand_table_clearance

    implementation = Path(__file__).read_bytes()
    required = {
        "source_result",
        "source_sha256",
        "time_s",
        "loaded_response_fraction",
        "maximum_joint_change_rad",
        "nonpad_clearance_m",
        "target_gap_m",
        "maximum_evaluations",
        "deadline_s",
        "output",
    }
    if set(config) - {"response_time_s"} != required or not (
        0 <= config["loaded_response_fraction"] <= 1
        and 0 < config["maximum_joint_change_rad"] <= 0.35
        and 0.0005 <= config["nonpad_clearance_m"] <= 0.003
        and -0.0005 <= config["target_gap_m"] <= 0
        and 1 <= config["maximum_evaluations"] <= 100
        and 0 < config["deadline_s"] <= 900
    ):
        raise ValueError("Explicit bounded loaded closure planning settings required")
    path = Path(config["source_result"])
    if digest(path) != config["source_sha256"]:
        raise ValueError("Pinned closure observation changed")
    source = json.loads(path.read_text())
    if (
        source.get("schema") != "real2sim2real.arm-replay-pilot/1"
        or source.get("physics_run") is not True
        or not source.get("continuous_protocol")
    ):
        raise ValueError("Continuous loaded arm physics required")
    for name, key in (("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")):
        if digest(path.with_name(name)) != source[key]:
            raise ValueError("Loaded closure artifacts changed")
    audit_path = Path(source["config"]["audit"])
    if digest(audit_path) != source["config"]["audit_sha256"]:
        raise ValueError("Pinned arm audit changed")
    audit = json.loads(audit_path.read_text())
    for name, sha in audit["sources"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Loaded planning source changed")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    if source["config"].get("finger_seating_plan") or parent["continuous_protocol"].get(
        "closing_wrist_offsets"
    ):
        raise ValueError("Use the frozen unmodified acquisition as the planning anchor")
    model = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    data = mujoco.MjData(model)
    response = None
    with np.load(path.with_name("states.npz"), allow_pickle=False) as states:
        i = int(np.argmin(abs(states["timestamps_s"] - config["time_s"])))
        data.qpos[:] = states["qpos"][i]
        if "response_time_s" in config:
            if source.get("motion_condition") != "stationary_hold":
                raise ValueError(
                    "Loaded motor calibration requires a stationary source"
                )
            time = config["response_time_s"]
            if (
                not np.isfinite(time)
                or not 0 <= time <= states["timestamps_s"][-1] + 0.001
            ):
                raise ValueError("Loaded response time must be inside the saved trial")
            j = int(np.argmin(abs(states["timestamps_s"] - time)))
            response_row = source["records"][j]
            if (
                response_row["phase"] != "hold"
                or response_row["table_contacts"] != 0
                or any(
                    not any(
                        p["loaded"]
                        for k, p in response_row["contact_patch_motion"]["pads"].items()
                        if k.startswith(role + ":")
                    )
                    for role in ("primary", "secondary")
                )
            ):
                raise ValueError("Suspended bilateral loaded response required")
            response = mujoco.MjData(model)
            response.qpos[:], response.qvel[:], response.ctrl[:] = (
                states["qpos"][j],
                states["qvel"][j],
                states["ctrl"][j],
            )
            mujoco.mj_forward(model, response)
    row = source["records"][i]
    if (
        abs(row["time_s"] - source["continuous_protocol"]["acquisition_end_s"]) > 0.03
        or row["table_contacts"] <= 0
    ):
        raise ValueError("Table-supported acquisition-completion observation required")
    mujoco.mj_fwdPosition(model, data)
    original = data.qpos.copy()
    roots = {a["name"]: model.body(a["root"]).id for a in audit["config"]["arms"]}
    _, _, mesh = load_scene(parent["configuration"])
    motors = [
        a for a in range(model.nu) if not model.actuator(a).name.startswith("pilot__")
    ]
    live = ArmFingerFeedback(model, data, parent, roots, mesh, motors)
    arm_names = [n for a in audit["config"]["arms"] for n in a["joints"]]
    arm_joints = np.array([model.joint(n).id for n in arm_names])
    arm_ids = model.jnt_qposadr[arm_joints]
    arm_start = original[arm_ids].copy()
    arm_lo = np.maximum(model.jnt_range[arm_joints, 0], arm_start - 0.03)
    arm_hi = np.minimum(model.jnt_range[arm_joints, 1], arm_start + 0.03)
    fingers = np.concatenate([p["qids"] for p in live.servo.pads])
    limits = np.concatenate([p["limits"] for p in live.servo.pads])
    nominal_by_id = {
        int(q): float(v)
        for h in live.hands
        for q, v in zip(h["qids"], h["waypoints"][2])
    }
    nominal = np.array([nominal_by_id[int(q)] for q in fingers])
    deflection = config["loaded_response_fraction"] * (nominal - original[fingers])
    response_evidence = dict(
        mode="nominal-minus-actual closure displacement", time_s=row["time_s"]
    )
    if response is not None:
        control_by_id = dict(zip(live.qids, live.actuators))
        bias_shift = {}
        for pad in live.servo.pads:
            for q, dof in zip(pad["qids"], pad["dofs"]):
                a = control_by_id[q]
                bias_shift[int(q)] = (
                    float(response.qfrc_bias[dof] / model.actuator_gainprm[a, 0])
                    if live.support_bias and pad["role"] == "secondary"
                    else 0.0
                )
        deflection = response_deflection(
            [response.ctrl[control_by_id[q]] for q in fingers],
            response.qpos[fingers],
            [bias_shift[int(q)] for q in fingers],
            config["loaded_response_fraction"],
        )
        response_evidence = dict(
            mode="loaded motor command-minus-actual displacement with existing support bias removed",
            time_s=response_row["time_s"],
            source_phase=response_row["phase"],
            velocity_l2_rad_s=float(
                np.linalg.norm(
                    response.qvel[np.concatenate([p["dofs"] for p in live.servo.pads])]
                )
            ),
            observed_forces_n={
                k: p["force_n"]
                for k, p in response_row["contact_patch_motion"]["pads"].items()
            },
        )
    lo, hi = command_bounds(
        nominal, deflection, limits, config["maximum_joint_change_rad"]
    )
    obj = model.body("task_object").id
    objects = {
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == obj and model.geom_contype[g]
    }
    object_pose = np.eye(4)
    object_pose[:3, :3], object_pose[:3, 3] = (
        data.xmat[obj].reshape(3, 3),
        data.xpos[obj],
    )
    object_rotation = object_pose[:3, :3]
    world_mesh = mesh.copy()
    world_mesh.apply_transform(object_pose)
    initial_wrists = {}
    for role, root in roots.items():
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.xmat[root].reshape(3, 3), data.xpos[root]
        initial_wrists[role] = pose
    hands = set().union(*(subtree_geoms(model, root) for root in roots.values()))
    arm_bodies = {
        model.body(n).id for a in audit["config"]["arms"] for n in a["collision_bodies"]
    }
    arms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bodies}
    robot = hands | arms
    nonpads = sorted(
        set().union(*(set(nonpad_chain_geoms(model, p)) for p in live.servo.pads))
    )
    query = hands & {
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    }
    model.geom_margin[list(query)] = np.maximum(model.geom_margin[list(query)], 0.03)
    command_data = mujoco.MjData(model)

    def sample(x):
        data.qpos[:] = original
        data.qpos[arm_ids] = arm_start + 0.03 * x[: len(arm_ids)]
        data.qpos[fingers] = x[len(arm_ids) :] - deflection
        mujoco.mj_fwdPosition(model, data)
        wrist = {}
        for role, root in roots.items():
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = data.xmat[root].reshape(3, 3), data.xpos[root]
            wrist[role] = offsets(initial_wrists[role], pose, object_rotation)
        gaps = [
            contact_witness(data, p["gid"], objects, 0.03)[0] for p in live.servo.pads
        ]
        other_gaps = [contact_witness(data, g, objects, 0.03)[0] for g in nonpads]
        collision = environment_excess(model, data, robot, arms, objects)
        command_data.qpos[:] = data.qpos
        command_data.qpos[fingers] = x[len(arm_ids) :]
        mujoco.mj_fwdPosition(model, command_data)
        commanded_collision = environment_excess(
            model, command_data, robot, arms, objects
        )
        return (
            wrist,
            np.array(gaps),
            np.array(other_gaps),
            collision,
            commanded_collision,
        )

    def residual(x):
        wrist, gaps, others, collision, commanded_collision = sample(x)
        bounds, regularization = [], []
        for w in wrist.values():
            bounds.extend(
                [
                    10 * max(0, np.linalg.norm(w["translation"]) - 0.0059),
                    max(0, np.linalg.norm(w["rotation"]) - 0.059),
                ]
            )
            regularization.extend(
                0.00003 * np.r_[w["translation"] / 0.006, w["rotation"] / 0.06]
            )
        return np.r_[
            gaps - config["target_gap_m"],
            5 * np.maximum(0, config["nonpad_clearance_m"] - others),
            10 * collision,
            10 * commanded_collision,
            bounds,
            regularization,
            0.0005 * (x[len(arm_ids) :] - nominal),
        ]

    seed = np.r_[np.zeros(len(arm_ids)), np.clip(nominal, lo + 1e-9, hi - 1e-9)]
    fit = least_squares(
        residual,
        seed,
        bounds=(
            np.r_[(arm_lo - arm_start) / 0.03, lo],
            np.r_[(arm_hi - arm_start) / 0.03, hi],
        ),
        max_nfev=config["maximum_evaluations"],
        diff_step=1e-4,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    wrist, gaps, others, collision, commanded_collision = sample(fit.x)
    closing = dict(
        vectors_object_m={k: w["translation"].tolist() for k, w in wrist.items()},
        rotation_vectors_object_rad={
            k: w["rotation"].tolist() for k, w in wrist.items()
        },
    )
    try:
        validate(closing)
        wrist_valid = True
    except ValueError:
        wrist_valid = False
    reports = []
    cursor = len(arm_ids)
    for pad, gap in zip(live.servo.pads, gaps):
        _, segment = contact_witness(data, pad["gid"], objects, 0.03)
        near, distance, faces = trimesh.proximity.closest_point(
            world_mesh, [segment[3:]]
        )
        count = len(pad["qids"])
        reports.append(
            dict(
                label=pad["role"] + ":" + pad["group"],
                geom=model.geom(pad["gid"]).name,
                joint_names=[
                    model.joint(int(model.dof_jntid[d])).name for d in pad["dofs"]
                ],
                nominal_closed_rad=[nominal_by_id[int(q)] for q in pad["qids"]],
                closed_rad=fit.x[cursor : cursor + count].tolist(),
                predicted_loaded_rad=data.qpos[pad["qids"]].tolist(),
                measured_response_rad=deflection[
                    cursor - len(arm_ids) : cursor - len(arm_ids) + count
                ].tolist(),
                gap_after_m=float(gap),
                source_mesh_projection_m=float(distance[0]),
                point_body_m=(
                    data.xmat[pad["body"]].reshape(3, 3).T
                    @ (segment[:3] - data.xpos[pad["body"]])
                ).tolist(),
                point_object_m=(
                    object_rotation.T @ (near[0] - object_pose[:3, 3])
                ).tolist(),
                normal_object=(
                    object_rotation.T @ world_mesh.face_normals[faces[0]]
                ).tolist(),
                witness_distance_error_m=abs(
                    abs(float(gap)) - float(np.linalg.norm(segment[:3] - segment[3:]))
                ),
            )
        )
        cursor += count
    clearance = hand_table_clearance(model, data, hands, model.geom("table").id)
    accepted = bool(
        wrist_valid
        and max(abs(gaps - config["target_gap_m"])) <= 0.001
        and min(others) >= config["nonpad_clearance_m"] - 0.0001
        and max(collision, default=0) <= 1e-8
        and max(commanded_collision, default=0) <= 1e-8
        and clearance >= 0.005
        and all(
            p["source_mesh_projection_m"] <= 0.003
            and p["witness_distance_error_m"] <= 0.00001
            for p in reports
        )
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "closing_grasp_plan.py").write_bytes(implementation)
    write_json(
        out / "plan.json",
        dict(
            schema="real2sim2real.closing-grasp-plan/1",
            config=config,
            source_sha256=digest(path),
            source_time_s=row["time_s"],
            source_success=source["success"],
            source_rejected_for=source["rejected_for"],
            geometry_accepted=accepted,
            closing_wrist_offsets=closing,
            pads=reports,
            minimum_nonpad_gap_m=float(min(others)),
            minimum_hand_table_gap_m=clearance,
            maximum_environment_penetration_excess_m=float(max(collision, default=0)),
            maximum_command_penetration_excess_m=float(
                max(commanded_collision, default=0)
            ),
            command_and_predicted_pose_checked=True,
            evaluations=int(fit.nfev),
            solver_message=str(fit.message),
            empirical_response_is_approximation=True,
            response_evidence=response_evidence,
            physical_retention_validated=False,
            object_fixed_only_in_kinematic_proposal=True,
            query_margins_exported=False,
            implementation_sha256=digest(out / "closing_grasp_plan.py"),
        ),
    )
    print(
        json.dumps(
            dict(
                geometry_accepted=accepted,
                evaluations=int(fit.nfev),
                maximum_pad_gap_error_m=float(max(abs(gaps - config["target_gap_m"]))),
                minimum_nonpad_gap_m=float(min(others)),
                closing_wrist_offsets=closing,
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        return run(config)
    from real2sim2real.bounded_process import run as bounded

    return bounded(
        [
            sys.executable,
            "-m",
            __spec__.name,
            "--config",
            str(args.config.resolve()),
            "--worker",
        ],
        timeout=min(config["deadline_s"], 900),
    )


if __name__ == "__main__":
    main()
