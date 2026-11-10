"""Bounded contact/posture proposals and matched free-object physics trials."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def local_name(name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in (".", "..")
    ):
        raise ValueError("A local nonempty experiment name is required")
    return name


def rotation_margin(model, qpos, prefix):
    """Signed distance to the nearest actual angular joint limit, in radians."""
    values = np.asarray(qpos, float)
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] != model.nq
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite nonempty generalized-position samples required")
    result = {}
    for axis in ("rx", "ry", "rz"):
        j = model.joint(prefix + axis)
        if not model.jnt_limited[j.id]:
            raise ValueError("Limited wrist joint required")
        q = values[:, j.qposadr[0]]
        lo, hi = model.jnt_range[j.id]
        result[axis] = float(np.minimum(q - lo, hi - q).min())
    return result


def select_candidate(groups):
    """Only all-case passes qualify; never promote a merely less-bad grasp."""
    passed = [
        g
        for g in groups
        if g["candidate_variant"]
        and g["trials"]
        and all(t["success"] for t in g["trials"])
    ]
    return (
        min(passed, key=lambda g: max(t["hold_slip_m"] for t in g["trials"]))["name"]
        if passed
        else None
    )


def source(config):
    from .object_grasp_pilot import load_plan

    spec = config["posture_search"]
    p = Path(spec["parent_trial"])
    if digest(p) != spec["parent_trial_sha256"]:
        raise ValueError("Pinned baseline changed")
    report = json.loads(p.read_text())
    if (
        not report["success"]
        or digest(p.parent / "states.npz") != report["states_sha256"]
    ):
        raise ValueError("Successful unchanged baseline and states required")
    if report["candidate"] not in load_plan(config)["candidates"]:
        raise ValueError("Baseline differs from the retained plan")
    for key in (
        "robot",
        "support_robot",
        "physics",
        "planning",
        "task",
        "support_controller",
        "clearance_feedback",
        "orientation_feedback",
        "contact_feedback",
        "finger_closure",
        "finger_surface_feedback",
    ):
        if config.get(key) != report["configuration"].get(key):
            raise ValueError(
                "Posture search must preserve scene, robots and baseline feedback"
            )
    for key, value in report["configuration"]["rollout"].items():
        if key != "hold_s" and config["rollout"][key] != value:
            raise ValueError("Posture search must preserve approach, closure and lift")
    if config["rollout"]["hold_s"] < report["configuration"]["rollout"]["hold_s"]:
        raise ValueError("Cannot shorten baseline hold")
    return report


def plan(config):
    import mujoco
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_wrist_frame,
        set_root_frame,
    )
    from real2sim2real.retargeting.contact_layout import contact_targets
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        fit_contacts,
        contact_penetrations,
    )

    spec = config["posture_search"]
    parent = source(config)
    variants = spec["variants"]
    if not 1 <= len(variants) <= 6 or not 1 <= spec["max_nfev"] <= 400:
        raise ValueError("Bounded posture search required")
    names = [local_name(v["name"]) for v in variants]
    if len(set(names)) != len(names) or set(names) & {
        "retained-baseline",
        "shared-baseline",
    }:
        raise ValueError("Distinct nonbaseline names required")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", config)
    original = load_plan(config)
    baseline = parent["candidate"]
    rows = [deepcopy(baseline)]
    mapping = {}
    for n, variant in enumerate(variants):
        model, data, mesh = load_scene(config)
        candidate = deepcopy(baseline)
        candidate.update(
            candidate_id=1000 + n, posture_variant=variant, rejected_for=[]
        )
        for stale in ("approach_search",):
            candidate.pop(stale, None)
        set_wrist_frame(
            model,
            data,
            config,
            candidate["base_position_m"],
            candidate["base_rotation"],
        )
        support = candidate["support"]
        set_root_frame(
            model,
            data,
            support["root_body"],
            support["fixture_prefix"],
            support["base_position_m"],
            support["base_rotation"],
        )
        mujoco.mj_resetData(model, data)
        idx = robot_indices(model, config["robot"]["root_body"])
        left = robot_indices(model, support["root_body"])
        data.qpos[left["q"]] = support["q_pregrasp"]
        objq = model.joint("task_object_free").qposadr[0]
        data.qpos[objq : objq + 7] = original["settling"]["object_qpos"]
        world = mesh.copy()
        world.apply_transform(original["settling"]["settled_world_from_object"])
        targets = contact_targets(
            world,
            candidate["pair"],
            variant["finger_offsets_m"],
            direction=candidate["thumb_side_direction"],
        )
        if targets is None:
            candidate.update(accepted=False, rejected_for=["missing_contact_surface"])
        else:
            fit = fit_contacts(
                model,
                data,
                config["robot"]["root_body"],
                config["robot"]["sites"],
                targets,
                np.asarray(candidate["q_grasp"]),
                max_nfev=spec["max_nfev"],
                joint_bounds=variant["joint_bounds"],
            )
            opened = fit["q"].copy()
            for i, name in enumerate(candidate["joint_names"]):
                if name.endswith(("MCP_FE", "PIP", "DIP", "IP", "thumb_CMC_FE")):
                    opened[i] -= 0.25
            limits = model.jnt_range[idx["joints"]]
            opened = np.clip(opened, limits[:, 0], limits[:, 1])
            xyz = [candidate["joint_names"].index("fixture_" + a) for a in "xyz"]
            pre = opened.copy()
            pre[xyz] += candidate["approach_offset_m"]
            peak = 0.0
            for t in np.linspace(
                0,
                1,
                int(np.ceil(np.linalg.norm(candidate["approach_offset_m"]) / 0.002))
                + 1,
            ):
                data.qpos[idx["q"]] = pre * (1 - t) + opened * t
                mujoco.mj_fwdPosition(model, data)
                peak = max(
                    peak, float(contact_penetrations(model, data, idx).max(initial=0))
                )
            reasons = []
            if max(fit["site_errors_m"]) > config["planning"]["contact_error_m"]:
                reasons.append("contact_target_error")
            if (
                max(fit["penetration_m"], default=0)
                > config["planning"]["max_penetration_m"]
            ):
                reasons.append("grasp_penetration")
            if peak > config["planning"]["max_penetration_m"]:
                reasons.append("approach_penetration")
            candidate.update(
                q_grasp=fit["q"].tolist(),
                q_open=opened.tolist(),
                q_pregrasp=pre.tolist(),
                targets_m=targets.tolist(),
                site_errors_m=fit["site_errors_m"].tolist(),
                max_penetration_m=float(max(fit["penetration_m"], default=0)),
                approach_max_penetration_m=peak,
                nfev=fit["nfev"],
                converged=fit["converged"],
                accepted=not reasons,
                rejected_for=reasons,
            )
            data.qpos[idx["q"]] = fit["q"]
            candidate["planned_wrist_margin_rad"] = rotation_margin(
                model, data.qpos[None], "fixture_"
            )
        rows.append(candidate)
        mapping[variant["name"]] = candidate["candidate_id"]
        print(variant["name"], candidate["rejected_for"], flush=True)
        write_json(root / "planning-progress.json", dict(candidates=rows))
    report = dict(
        schema="real2sim2real.grasp-posture-plan/1",
        candidates=rows,
        settling=original["settling"],
        scene_receipt_sha256=original["scene_receipt_sha256"],
        parent_trial_sha256=spec["parent_trial_sha256"],
        variants=mapping,
        protocol_sha256=digest(root / "protocol.json"),
        physical_success_validated=False,
    )
    (root / "planning").mkdir()
    write_json(root / "planning/plan.json", report)
    return report


def compare(config):
    from .object_grasp_pilot import trial, load_scene
    from .grasp_robustness import summary
    from .wrist_coordination import hold_world_motion

    parent = source(config)
    spec = config["posture_search"]
    root = Path(spec["output"])
    plan = json.loads((root / "planning/plan.json").read_text())
    if json.loads((root / "protocol.json").read_text()) != config or plan[
        "protocol_sha256"
    ] != digest(root / "protocol.json"):
        raise ValueError("Posture protocol changed")
    cases = spec["cases"]
    names = [local_name(c["name"]) for c in cases]
    if not 1 <= len(cases) <= 6 or len(set(names)) != len(names):
        raise ValueError("Bounded distinct material cases required")
    output = root / "comparison"
    output.mkdir(exist_ok=False)
    current = deepcopy(config)
    current["planning_directory"] = str(root / "planning")
    candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    variants = {
        "retained-baseline": parent["candidate"]["candidate_id"],
        "shared-baseline": parent["candidate"]["candidate_id"],
        **plan["variants"],
    }
    model, _, _ = load_scene(config)
    groups = []
    for name, ident in variants.items():
        candidate = candidates[ident]
        if not candidate["accepted"]:
            continue
        current["object_frame_coordination"] = (
            parent["controller"].get("object_frame_coordination")
            if name == "retained-baseline"
            else config["object_frame_coordination"]
        )
        rows = []
        for case in cases:
            directory = name + "--" + case["name"]
            result = trial(
                current,
                candidate,
                output / directory,
                mass_scale=case["mass_scale"],
                friction=case["friction"],
            )
            states = np.load(output / directory / "states.npz")
            row = summary(result, directory)
            row.update(hold_world_motion(result["records"]))
            row["wrist_margin_rad"] = {
                p: rotation_margin(model, states["qpos"], p)
                for p in ("fixture_", config["support_robot"]["fixture_prefix"])
            }
            row["result_sha256"] = digest(output / directory / "result.json")
            rows.append(row)
            print(directory, result["success"], result["rejected_for"], flush=True)
        groups.append(
            dict(
                name=name,
                candidate_id=ident,
                candidate_variant=ident != parent["candidate"]["candidate_id"],
                trials=rows,
            )
        )
        write_json(output / "progress.json", dict(groups=groups))
    selected = select_candidate(groups)
    report = dict(
        schema="real2sim2real.grasp-posture-comparison/1",
        groups=groups,
        selected_variant=selected,
        decision="development_pass_requires_frozen_validation"
        if selected
        else "retain_baseline_no_all_case_pass",
        plan_sha256=digest(root / "planning/plan.json"),
        thresholds_relaxed=False,
        policy_trained=False,
    )
    write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["plan", "compare"], required=True)
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
            timeout=config["posture_search"]["deadlines_s"][args.phase],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide GPU for CPU posture experiments")
    return (plan if args.phase == "plan" else compare)(config)


if __name__ == "__main__":
    main()
