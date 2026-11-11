"""Audit a retained primary and replacement support without refitting either.

Simultaneous linear pregrasp/open/closed paths are a geometric hypothesis, not
loaded tracking, continuous collision certification or a new controller pass.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def paired_waypoints(hands):
    if len(hands) != 2:
        raise ValueError("Exactly two hands required")
    arrays = []
    names = []
    for hand in hands:
        names.extend(hand["joint_names"])
        q = np.asarray([hand[k] for k in ("q_pregrasp", "q_open", "q_closed")], float)
        if q.shape != (3, len(hand["joint_names"])) or not np.isfinite(q).all():
            raise ValueError("Complete finite waypoints in native joint order required")
        arrays.append(q)
    if len(set(names)) != len(names):
        raise ValueError("Hands must own distinct joints")
    return np.concatenate(arrays, axis=1)


def pinned_json(receipt):
    path = Path(receipt["path"])
    if digest(path) != receipt["sha256"]:
        raise ValueError("Pinned acquisition input changed")
    return json.loads(path.read_text())


def run(config):
    import mujoco
    from .object_grasp_pilot import load_scene, load_plan, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import contact_patch_samples
    from real2sim2real.retargeting.paired_contacts import paired_wrench_screen

    proposal = pinned_json(config["plan"])
    parent = pinned_json(proposal["config"]["source_trial"])
    index = config["candidate_index"]
    if type(index) is not int or not 0 <= index < len(proposal["candidates"]):
        raise ValueError("A declared candidate index is required")
    replacement = proposal["candidates"][index]
    if not proposal.get(
        "retained_primary_and_object_state_verified"
    ) or not replacement.get("geometry_candidate"):
        raise ValueError("Require the state-preserving geometric preflight")
    settings = parent["configuration"]
    model, data, mesh = load_scene(settings)
    settled = load_plan(settings)["settling"]
    mesh.apply_transform(np.asarray(settled["settled_world_from_object"]))
    hands = [parent["candidate"], replacement]
    waypoints = paired_waypoints(hands)
    robots = [settings["robot"], settings["support_robot"]]
    indices = [robot_indices(model, robot["root_body"]) for robot in robots]
    for hand, idx in zip(hands, indices):
        if hand["joint_names"] != [model.joint(int(j)).name for j in idx["joints"]]:
            raise ValueError("Native joint ownership/order changed")
    combined = {
        key: np.concatenate([idx[key] for idx in indices])
        for key in ("joints", "q", "dofs", "geoms")
    }
    obj = model.joint("task_object_free").qposadr[0]
    data.qpos[obj : obj + 7] = settled["object_qpos"]
    data.qpos[combined["q"]] = waypoints[0]
    initial = data.qpos.copy()
    for hand, robot in zip(hands, robots):
        set_root_frame(
            model,
            data,
            robot["root_body"],
            robot.get("fixture_prefix", "fixture_"),
            hand["base_position_m"],
            hand["base_rotation"],
            preserve_state=True,
        )
        if not np.array_equal(data.qpos, initial):
            raise ValueError("Root placement changed acquisition or object state")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        Path(inspect.getfile(load_scene)),
        Path(inspect.getfile(path_audit)),
        Path(inspect.getfile(robot_indices)),
        Path(inspect.getfile(contact_patch_samples)),
        Path(inspect.getfile(paired_wrench_screen)),
    ]
    hashes = {str(p): digest(p) for p in paths}
    snapshots = output / "implementation"
    snapshots.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshots / path.name)
    phases = {}
    for i, name in enumerate(("approach", "closure")):
        phases[name] = path_audit(model, data, combined, waypoints[i : i + 2])
    if not np.array_equal(data.qpos, initial):
        raise ValueError("Collision queries mutated the retained acquisition state")
    data.qpos[combined["q"]] = waypoints[-1]
    mujoco.mj_fwdPosition(model, data)
    patches = [
        contact_patch_samples(model, data, hand["pad_plan"]["frames"], mesh)
        for hand in hands
    ]
    wrench = paired_wrench_screen(
        *patches,
        mesh.center_mass,
        settings["physics"]["mass_kg"],
        settings["physics"]["friction"],
        proposal["config"]["maximum_normal_force_n"],
    )
    reasons = []
    limit = settings["planning"]["max_penetration_m"]
    for name, phase in phases.items():
        if phase["max_penetration_m"] > limit:
            reasons.append(name + "_penetration")
        if not phase["within_joint_limits"]:
            reasons.append(name + "_joint_limits")
    if not all(p["all_pads_observed"] for p in patches):
        reasons.append("finite_pad_coverage")
    if not wrench["feasible"]:
        reasons.append("finite_patch_gravity_balance")
    # Recheck inputs as well as code; this is an audit, never a baseline rewrite.
    pinned_json(config["plan"])
    pinned_json(proposal["config"]["source_trial"])
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError("Audit implementation changed")
    result = dict(
        schema="paired-acquisition-audit/1",
        config=config,
        phases=phases,
        maximum_penetration_m=limit,
        patches=patches,
        paired_gravity=wrench,
        geometric_path_pass=not reasons,
        rejected_for=reasons,
        source_controller_success=parent.get("success"),
        initial_state_preserved=True,
        implementation_sources=hashes,
        acquisition=dict(
            approach_s=settings["rollout"]["approach_s"],
            close_s=settings["rollout"]["close_s"],
            interpolation="simultaneous linear pregrasp/open/closed; timing and loaded response not validated",
        ),
        dynamic_success_validated=False,
        control_admitted=False,
        arm_reachability_validated=False,
        baseline_replaced=False,
    )
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Require a finite audit deadline up to 900 seconds")
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
