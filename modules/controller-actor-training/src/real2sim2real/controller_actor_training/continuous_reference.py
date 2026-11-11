"""Plan acquisition on the capture clock, without a separate robot lift.

Outputs are kinematic command proxies. Fresh whole-arm path admission and free
object dynamics are required; no generated state is presented as an observation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json


def reference_goal(reference, elapsed, anchor):
    """Retain captured world-axis pose increments at an explicit resting anchor."""
    translation, rotation = reference.sample(elapsed)
    goal = np.array(anchor, float, copy=True)
    goal[:3, 3] += translation
    goal[:3, :3] = rotation @ goal[:3, :3]
    return goal


def table_command(goal, vertices, table_z, maximum_adjustment_m):
    """Project the *command* above the table; retain raw reference for scoring."""
    goal, vertices = np.asarray(goal, float), np.asarray(vertices, float)
    if (
        goal.shape != (4, 4)
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(goal).all()
        or not np.isfinite(vertices).all()
        or not np.isfinite([table_z, maximum_adjustment_m]).all()
        or not 0 <= maximum_adjustment_m <= 0.01
    ):
        raise ValueError(
            "Finite pose, nonempty metric mesh and <=10 mm command budget required"
        )
    points = vertices @ goal[:3, :3].T + goal[:3, 3]
    clearance = float(points[:, 2].min() - table_z)
    adjustment = max(0.0, -clearance)
    if not np.isfinite(adjustment) or adjustment > maximum_adjustment_m:
        raise ValueError("Raw reference needs an excessive table command correction")
    command = goal.copy()
    command[2, 3] += adjustment
    return command, clearance, adjustment


def phase_at(
    time, approach_s, close_s, clearance, minimum_clearance, carried, duration
):
    if time < approach_s:
        return "approach"
    if time < approach_s + close_s:
        return "close"
    if time > duration:
        return "settle"
    if clearance >= minimum_clearance:
        return "hold"
    return "return" if carried else "lift"


def evaluate_continuous(records, protocol, table_z, *, stationary=False):
    """Acquisition-anchored retention and independently declared table return."""
    from .grasp_feedback import relative_translation_drift

    retained = [r for r in records if r["time_s"] >= protocol["acquisition_end_s"]]
    if not retained:
        return {}, ["missing_continuous_retention"]
    drift = {}
    for prefix, key in (("", "hold_slip_m"), ("support_", "support_wrist_drift_m")):
        drift[key] = relative_translation_drift(
            [r["object_position_m"] for r in retained],
            [r[prefix + "wrist_position_m"] for r in retained],
            [r[prefix + "wrist_rotation"] for r in retained],
        )
    penetration = max(0.0, max(table_z - r["minimum_object_height_m"] for r in records))
    final = [r for r in records if r["time_s"] >= protocol["capture_duration_s"]]
    result = dict(
        **drift,
        retention_anchor_time_s=retained[0]["time_s"],
        maximum_object_table_penetration_m=penetration,
        return_evaluated=not stationary,
    )
    failures = ["object_table_penetration"] if penetration > 0.003 else []
    if not stationary:
        coverage = final[-1]["time_s"] - final[0]["time_s"] if final else 0.0
        contact = (
            float(np.mean([r["table_contacts"] > 0 for r in final])) if final else 0.0
        )
        final_clearance = max(
            (r["minimum_object_height_m"] - table_z for r in final),
            default=float("inf"),
        )
        result.update(
            final_table_contact_fraction=contact,
            final_maximum_clearance_m=final_clearance,
            final_settle_coverage_s=coverage,
        )
        if (
            contact < 0.8
            or final_clearance > 0.005
            or coverage < protocol["terminal_settle_s"] - 0.05
        ):
            failures.append("incomplete_table_return")
    return result, failures


def run(config):
    import mujoco
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from .arm_finger_feedback import finger_waypoint
    from .arm_finger_plan import bind
    from .guided_motion import load_reference
    from .object_grasp_pilot import load_scene, load_plan, set_root_frame

    implementation = Path(__file__).read_bytes()

    if set(config) != {
        "baseline_result",
        "baseline_sha256",
        "output",
        "sample_period_s",
        "terminal_settle_s",
        "maximum_table_adjustment_m",
        "deadline_s",
    }:
        raise ValueError("Explicit continuous-reference planning protocol required")
    if not (
        0.01 <= config["sample_period_s"] <= 0.025
        and 1 <= config["terminal_settle_s"] <= 2
        and 0 <= config["maximum_table_adjustment_m"] <= 0.01
        and 0 < config["deadline_s"] <= 900
    ):
        raise ValueError(
            "Continuous protocol exceeds sample, settling or adjustment budget"
        )
    path = Path(config["baseline_result"])
    if digest(path) != config["baseline_sha256"]:
        raise ValueError("Frozen baseline changed")
    baseline = json.loads(path.read_text())
    if (
        baseline.get("residual_adapter")
        or baseline.get("finger_control_mode") != "live_contact_force"
    ):
        raise ValueError("Use an uncorrected live-finger baseline")
    audit_path = Path(baseline["config"]["audit"])
    if digest(audit_path) != baseline["config"]["audit_sha256"]:
        raise ValueError("Source arm audit changed")
    audit = json.loads(audit_path.read_text())
    sources = dict(audit["sources"])
    sources.update({str(path): digest(path), str(audit_path): digest(audit_path)})
    for name, sha in sources.items():
        if digest(Path(name)) != sha:
            raise ValueError("Continuous planning source changed")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    if baseline["config"].get("finger_plan"):
        parent, finger_admission = bind(parent, baseline["config"]["finger_plan"], out)
        sources[finger_admission["plan"]] = finger_admission["sha256"]
    else:
        finger_admission = None
    base = deepcopy(parent["configuration"])
    reference = load_reference(base["reference_guidance"])
    sources[base["reference_guidance"]["path"]] = base["reference_guidance"]["sha256"]
    task = json.loads(Path(base["task"]).read_text())
    plan = load_plan(base)
    for source in (
        base["task"],
        Path(base["output"]) / "scene/scene.xml",
        Path(base["output"]) / "settling.json",
        Path(base["output"]) / base.get("planning_directory", "planning") / "plan.json",
    ):
        sources[str(source)] = digest(Path(source))
    anchor = np.asarray(plan["settling"]["settled_world_from_object"], float)
    model, data, mesh = load_scene(base)
    hands = [deepcopy(parent["candidate"]), deepcopy(parent["candidate"]["support"])]
    hands[0].update(root_body=base["robot"]["root_body"], fixture_prefix="fixture_")
    indices = []
    for hand in hands:
        set_root_frame(
            model,
            data,
            hand["root_body"],
            hand["fixture_prefix"],
            hand["base_position_m"],
            hand["base_rotation"],
        )
        indices.append(robot_indices(model, hand["root_body"]))
    approach, close = base["rollout"]["approach_s"], base["rollout"]["close_s"]
    if approach + close >= reference.elapsed[-1]:
        raise ValueError("Acquisition must finish within the capture")
    duration = float(reference.elapsed[-1] + config["terminal_settle_s"])
    times = np.r_[np.arange(0, duration, config["sample_period_s"]), duration]
    if len(times) > 1000:
        raise ValueError("Continuous plan exceeds the dense arm-audit budget")
    object_q = model.joint("task_object_free").qposadr[0]
    records, states, controls, adjustments = [], [], [], []
    carried = False
    for time in times:
        goal = reference_goal(reference, time, anchor)
        command, clearance, adjustment = table_command(
            goal,
            mesh.vertices,
            task["table"]["top_z_m"],
            config["maximum_table_adjustment_m"],
        )
        phase = phase_at(
            time,
            approach,
            close,
            clearance,
            task["goal"]["minimum_clearance_m"],
            carried,
            reference.elapsed[-1],
        )
        carried |= phase == "hold"
        if (
            time < approach + close
            and np.linalg.norm(goal[:3, 3] - anchor[:3, 3]) > 0.01
        ):
            raise ValueError(
                "Object moves before declared robot acquisition can finish"
            )
        data.qpos[:] = model.qpos0
        pose = anchor if time == 0 else command
        data.qpos[object_q : object_q + 3] = pose[:3, 3]
        data.qpos[object_q + 3 : object_q + 7] = Rotation.from_matrix(
            pose[:3, :3]
        ).as_quat()[[3, 0, 1, 2]]
        change = command @ np.linalg.inv(anchor)
        wrists = []
        for hand, idx in zip(hands, indices):
            data.qpos[idx["q"]] = finger_waypoint(
                time,
                hand["q_pregrasp"],
                hand["q_open"],
                hand["q_closed"],
                base["rollout"],
            )
            mujoco.mj_kinematics(model, data)
            root = idx["root"]
            wrist = np.eye(4)
            wrist[:3, 3], wrist[:3, :3] = data.xpos[root], data.xmat[root].reshape(3, 3)
            target = change @ wrist
            q = np.array(
                [
                    model.joint(hand["fixture_prefix"] + s).qposadr[0]
                    for s in ("x", "y", "z", "rx", "ry", "rz")
                ]
            )
            data.qpos[q[:3]] += target[:3, 3] - wrist[:3, 3]
            base_rotation = Rotation.from_quat(
                model.body_quat[root][[1, 2, 3, 0]]
            ).as_matrix()
            data.qpos[q[3:]] = Rotation.from_matrix(
                base_rotation.T @ target[:3, :3]
            ).as_euler("XYZ")
            wrists.append(target)
        record = dict(
            time_s=float(time),
            phase=phase,
            object_position_m=command[:3, 3].tolist(),
            object_rotation=command[:3, :3].tolist(),
            reference_position_m=goal[:3, 3].tolist(),
            reference_rotation=goal[:3, :3].tolist(),
            reference_elapsed_s=float(min(time, reference.elapsed[-1])),
            planned_reference_clearance_m=clearance,
            table_command_adjustment_m=adjustment,
        )
        for prefix, wrist in zip(("", "support_"), wrists):
            record[prefix + "wrist_position_m"] = wrist[:3, 3].tolist()
            record[prefix + "wrist_rotation"] = wrist[:3, :3].tolist()
        records.append(record)
        states.append(data.qpos.copy())
        controls.append(
            np.array(
                [
                    data.qpos[model.jnt_qposadr[int(model.actuator_trnid[a, 0])]]
                    for a in range(model.nu)
                ]
            )
        )
        adjustments.append(adjustment)
    if not carried:
        raise ValueError("Capture does not provide the required suspended segment")
    base["rollout"].update(lift_m=0.0, lift_s=0.0)
    protocol = dict(
        schema="real2sim2real.continuous-reference-protocol/1",
        capture_clock_origin_s=float(reference.times[0]),
        duration_s=duration,
        capture_duration_s=float(reference.elapsed[-1]),
        frame_ids=reference.frame_ids.tolist(),
        acquisition_end_s=approach + close,
        independent_lift_m=0.0,
        reference_start_s=0.0,
        initial_world_from_object=anchor.tolist(),
        original_capture_absolute_pose_reproduced=False,
        maximum_table_command_adjustment_m=max(adjustments),
        raw_reference_preserved=True,
        native_timing_preserved=True,
        terminal_settle_s=config["terminal_settle_s"],
        release_planned=False,
        phase_gates="50 mm clearance and load support on reference-defined suspended hold; raw pose error over full capture; drift anchored at acquisition completion",
        finger_plan_admission=finger_admission,
    )
    source = out / "source-trial"
    source.mkdir()
    (source / "continuous_reference.py").write_bytes(implementation)
    np.savez_compressed(
        source / "states.npz", qpos=states, ctrl=controls, timestamps_s=times
    )
    report = dict(
        schema="real2sim2real.continuous-reference-plan/1",
        configuration=base,
        candidate=parent["candidate"],
        records=records,
        continuous_protocol=protocol,
        success=False,
        rejected_for=["kinematic_plan_requires_new_path_and_dynamics"],
        states_sha256=digest(source / "states.npz"),
        physics_run=False,
        source_hashes=sources,
        implementation_sha256=digest(source / "continuous_reference.py"),
    )
    write_json(source / "result.json", report)
    write_json(out / "protocol.json", dict(config=config, **protocol))
    print(
        json.dumps(
            dict(
                samples=len(times),
                duration_s=duration,
                maximum_command_adjustment_m=max(adjustments),
            )
        ),
        flush=True,
    )
    return report


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
