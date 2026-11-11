"""Transfer a loaded coupled wrist/pad fit back to a fresh object-centric grasp.

Fit/open/preload, approach collision and gravity-wrench checks precede a fresh
unassisted fixture rollout. Source states are planning evidence, never a warm
start that teleports the object into a grasp.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from real2sim2real.artifacts import digest, write_json


def object_relative_pose(initial_object, loaded_object, loaded_wrist):
    poses = [
        np.asarray(p, float) for p in (initial_object, loaded_object, loaded_wrist)
    ]
    for p in poses:
        if (
            p.shape != (4, 4)
            or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1])
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(p[:3, :3]), 1)
        ):
            raise ValueError("Rigid finite metric poses required")
    return poses[0] @ np.linalg.inv(poses[1]) @ poses[2]


def run(config):
    import mujoco
    import trimesh
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_wrist_frame,
        set_root_frame,
        object_transform,
        trial,
    )
    from .loaded_finger_replan import rebase_wrist
    from .pad_grasp_planning import fit_candidate, path_audit
    from .retention_experiment import assess
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    pinned = {}
    for key in ("parent_trial", "loaded_fit"):
        path = Path(config[key])
        if digest(path) != config[key + "_sha256"]:
            raise ValueError("Pinned coupled-grasp input changed")
        pinned[key] = json.loads(path.read_text())
    parent, fitted = pinned["parent_trial"], pinned["loaded_fit"]
    if (
        not parent["success"]
        or not fitted["geometry_accepted"]
        or fitted.get("wrist_search") is None
    ):
        raise ValueError(
            "Require accepted source grasp and explicitly coupled geometry fit"
        )
    path = Path(fitted["config"]["parent_trial"])
    if digest(path) != fitted["config"]["parent_trial_sha256"]:
        raise ValueError("Loaded-fit source changed")
    source = json.loads(path.read_text())
    if digest(path.with_name("states.npz")) != source["states_sha256"]:
        raise ValueError("Loaded-fit states changed")
    if (
        digest(Path(config["parent_trial"]).with_name("states.npz"))
        != parent["states_sha256"]
    ):
        raise ValueError("Retained grasp states changed")
    for key in ("task", "robot", "physics"):
        if source["configuration"][key] != parent["configuration"][key]:
            raise ValueError(
                "Loaded fit and retained grasp must share object, robot and physics"
            )
    base = deepcopy(parent["configuration"])
    plan = load_plan(base)
    if parent["candidate"] not in plan["candidates"]:
        raise ValueError("Retained candidate is absent from its pinned plan")
    old, data, _ = load_scene(source["configuration"])
    c = source["candidate"]
    set_wrist_frame(
        old, data, source["configuration"], c["base_position_m"], c["base_rotation"]
    )
    s = c["support"]
    set_root_frame(
        old,
        data,
        s["root_body"],
        s["fixture_prefix"],
        s["base_position_m"],
        s["base_rotation"],
    )
    with np.load(path.with_name("states.npz"), allow_pickle=False) as archive:
        data.qpos[:] = archive["qpos"][fitted["config"]["sample_index"]]
    mujoco.mj_fwdPosition(old, data)
    loaded_object = object_transform(old, data)
    idx = robot_indices(old, base["robot"]["root_body"])
    names = [old.joint(int(j)).name for j in idx["joints"]]
    if names != fitted["joint_names"]:
        raise ValueError("Fitted joint ordering differs")
    rebase_wrist(old, data, base, idx, names)
    data.qpos[idx["q"]] = fitted["fitted_q"]
    mujoco.mj_fwdPosition(old, data)
    wrist = np.eye(4)
    wrist[:3, :3] = data.xmat[idx["root"]].reshape(3, 3)
    wrist[:3, 3] = data.xpos[idx["root"]]
    initial_object = np.asarray(plan["settling"]["settled_world_from_object"])
    initial_wrist = object_relative_pose(initial_object, loaded_object, wrist)
    transfer = initial_object @ np.linalg.inv(loaded_object)
    targets = trimesh.transform_points(fitted["target_world_m"], transfer)
    pads = [
        dict(
            geom=f["geom"],
            outward_body=f["declared_outward_body"],
            point_body_m=f["point_body_m"],
        )
        for f in fitted["pad_frames"]
    ]
    # Replace the legacy undirected flexion preload with a fitted pad closing path.
    base["rollout"]["close_offset_rad"] = 0
    variants = config["preloads_m"]
    if not 1 <= len(variants) <= 3 or len(set(variants)) != len(variants):
        raise ValueError("One to three explicit preload hypotheses required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    for filename in (
        "coupled_grasp_transfer.py",
        "pad_grasp_planning.py",
        "loaded_finger_replan.py",
    ):
        (out / "implementation" / filename).write_bytes(
            Path(__file__).with_name(filename).read_bytes()
        )
    candidates = []
    for i, preload in enumerate(variants):
        model, data, mesh = load_scene(base)
        candidate = deepcopy(parent["candidate"])
        set_wrist_frame(model, data, base, initial_wrist[:3, 3], initial_wrist[:3, :3])
        s = candidate["support"]
        set_root_frame(
            model,
            data,
            s["root_body"],
            s["fixture_prefix"],
            s["base_position_m"],
            s["base_rotation"],
        )
        objq = model.joint("task_object_free").qposadr[0]
        data.qpos[objq : objq + 7] = plan["settling"]["object_qpos"]
        left = robot_indices(model, s["root_body"])
        data.qpos[left["q"]] = s["q_pregrasp"]
        idx = robot_indices(model, base["robot"]["root_body"])
        seed = np.asarray(fitted["fitted_q"]).copy()
        for j, name in enumerate(names):
            if name.startswith("fixture_"):
                seed[j] = 0
        mesh.apply_transform(initial_object)
        base["pad_planning"] = dict(
            pads=pads,
            opening_gap_m=config["opening_gap_m"],
            preload_m=preload,
            max_normal_error_deg=25,
            max_normal_force_n=config["maximum_normal_force_n"],
        )
        fit, opened, _, extra, actual_targets = fit_candidate(
            model, data, base, mesh, targets, seed, idx, names.index("fixture_z")
        )
        xyz = [names.index("fixture_" + a) for a in "xyz"]
        approaches = np.asarray(
            config.get("approach_offsets_m", [candidate["approach_offset_m"]]), float
        )
        if (
            approaches.ndim != 2
            or approaches.shape[1] != 3
            or not 1 <= len(approaches) <= 8
            or not np.isfinite(approaches).all()
            or (np.linalg.norm(approaches, axis=1) > 0.15).any()
        ):
            raise ValueError(
                "One to eight finite approach offsets within 150 mm required"
            )
        paths = []
        for offset in approaches:
            pre = opened.copy()
            pre[xyz] += offset
            audit = path_audit(model, data, idx, [pre, opened, extra["q_closed"]])
            paths.append(
                dict(offset_m=offset.tolist(), q_pregrasp=pre.tolist(), **audit)
            )
        picked = min(
            paths, key=lambda p: (not p["within_joint_limits"], p["max_penetration_m"])
        )
        pre = np.asarray(picked["q_pregrasp"])
        candidate["approach_offset_m"] = picked["offset_m"]
        pad = extra["pad_plan"]
        pad["original_vertical_path"] = pad["path"]
        pad["approach_hypotheses"] = paths
        pad["path"] = {
            k: picked[k]
            for k in (
                "max_penetration_m",
                "samples",
                "within_joint_limits",
                "continuous_collision_certified",
            )
        }
        pad["closing_segment"] = path_audit(
            model, data, idx, [opened, extra["q_closed"]]
        )
        reasons = [
            r
            for r in pad["rejected_for"]
            if r not in ("pad_closing_path_penetration", "pad_path_joint_limits")
        ]
        if pad["path"]["max_penetration_m"] > base["planning"]["max_penetration_m"]:
            reasons.append("pad_closing_path_penetration")
        if not pad["path"]["within_joint_limits"]:
            reasons.append("pad_path_joint_limits")
        if fit["site_errors_m"].max() > base["planning"]["contact_error_m"]:
            reasons.append("pad_position_error")
        pad["rejected_for"] = reasons
        for key in (
            "approach_search",
            "contact_replan",
            "clearance_screen",
            "proposal_name",
        ):
            candidate.pop(key, None)
        candidate.update(
            candidate_id=4000 + i,
            coupled_fit_sha256=config["loaded_fit_sha256"],
            base_position_m=initial_wrist[:3, 3].tolist(),
            base_rotation=initial_wrist[:3, :3].tolist(),
            q_grasp=fit["q"].tolist(),
            q_open=opened.tolist(),
            q_pregrasp=pre.tolist(),
            targets_m=actual_targets.tolist(),
            site_errors_m=fit["site_errors_m"].tolist(),
            max_penetration_m=float(fit["penetration_m"].max(initial=0)),
            approach_max_penetration_m=pad["path"]["max_penetration_m"],
            score=float(max(fit["site_errors_m"]) + pad["path"]["max_penetration_m"]),
            accepted=not reasons,
            rejected_for=reasons,
            **extra,
        )
        candidates.append(candidate)
        print(
            "preload",
            preload,
            "rejected",
            reasons,
            "normal",
            pad["grasp_normal_errors_deg"],
            flush=True,
        )
        write_json(out / "planning-progress.json", dict(candidates=candidates))
    (out / "planning").mkdir()
    write_json(
        out / "planning/plan.json",
        dict(
            plan, candidates=candidates, accepted=sum(c["accepted"] for c in candidates)
        ),
    )
    rows = []
    for candidate, preload in zip(candidates, variants):
        if not candidate["accepted"]:
            continue
        settings = deepcopy(base)
        settings["pad_planning"]["preload_m"] = preload
        settings["planning_directory"] = str(out / "planning")
        result = trial(
            settings, candidate, out / f"candidate-{candidate['candidate_id']}"
        )
        rows.append(
            dict(
                candidate_id=candidate["candidate_id"],
                success=result["success"],
                rejected_for=result["rejected_for"],
                hold_slip_m=result["hold_slip_m"],
                **assess(result, config["fidelity_limits"]),
            )
        )
        write_json(out / "physics-progress.json", dict(trials=rows))
        print(rows[-1], flush=True)
    report = dict(
        schema="real2sim2real.coupled-grasp-transfer/1",
        config=config,
        candidates=[
            dict(
                candidate_id=c["candidate_id"],
                accepted=c["accepted"],
                rejected_for=c["rejected_for"],
            )
            for c in candidates
        ],
        trials=rows,
        baseline_replaced=False,
        arm_control_validated=False,
        implementation_sha256=digest(out / "implementation/coupled_grasp_transfer.py"),
    )
    write_json(out / "comparison.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
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
            timeout=900,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only experiment required")
    run(json.loads(a.config.read_text()))


if __name__ == "__main__":
    main()
