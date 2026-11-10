"""Fresh simultaneous-grasp physics from an admitted bilateral contact plan."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def assess_motion_condition(result, condition, limits):
    """A stationary diagnostic can never qualify demonstrated-motion control."""
    if condition == "demonstrated":
        from .retention_experiment import assess

        return assess(result, limits)
    if condition != "stationary_diagnostic":
        raise ValueError("Unknown bilateral motion condition")
    complete = (
        result["simulation_final_time_s"]
        >= result["simulation_requested_duration_s"] - 1e-6
    )
    return dict(
        combined_pass=False,
        reference_pass=None,
        fidelity=None,
        diagnostic_hold_pass=bool(result["success"] and complete),
        complete=complete,
    )


def run(config):
    import numpy as np
    from .object_grasp_pilot import load_plan, trial, load_scene, set_root_frame
    from .bilateral_grasp import rebase_closed_wrist
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    path = Path(config["plan"])
    if digest(path) != config["plan_sha256"]:
        raise ValueError("Pinned bilateral plan changed")
    plan = json.loads(path.read_text())
    source = Path(plan.get("parent_trial", plan["config"].get("parent_trial", "")))
    if digest(source) != plan.get(
        "parent_trial_sha256", plan["config"].get("parent_trial_sha256")
    ):
        raise ValueError("Retained parent changed")
    parent = json.loads(source.read_text())
    base = deepcopy(parent["configuration"])
    old_plan = load_plan(base)
    if (
        not parent["success"]
        or digest(source.with_name("states.npz")) != parent["states_sha256"]
    ):
        raise ValueError("Unchanged passing retained parent required")
    ids = config["candidate_ids"]
    if not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise ValueError("One to four distinct admitted candidates required")
    source_candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    if not all(
        i in source_candidates
        and source_candidates[i]["accepted"]
        and source_candidates[i]["gravity_wrench"]["feasible"]
        for i in ids
    ):
        raise ValueError("All requested candidates must pass paired contact admission")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    for filename in (
        "bilateral_evaluate.py",
        "bilateral_grasp.py",
        "support_control.py",
        "object_grasp_pilot.py",
    ):
        (out / "implementation" / filename).write_bytes(
            Path(__file__).with_name(filename).read_bytes()
        )
    base["rollout"]["close_offset_rad"] = 0
    base["support_controller"]["mode"] = "simultaneous_grasp"
    profile = config.get("controller_profile", "inherited")
    if profile not in ("inherited", "rigid_bilateral"):
        raise ValueError("Unknown simultaneous controller profile")
    if profile == "rigid_bilateral":
        # Independent handle-retention corrections can shear an opposing pair.
        # Both wrists retain common lift and demonstrated rigid object motion.
        base.pop("orientation_feedback", None)
        base.pop("grasp_retention", None)
        base["support_controller"].pop("load_feedback", None)

    motion_condition = config.get("motion_condition", "demonstrated")
    if motion_condition not in ("demonstrated", "stationary_diagnostic"):
        raise ValueError("Unknown bilateral motion condition")
    if motion_condition == "stationary_diagnostic":
        base.pop("reference_guidance", None)
        base.pop("grasp_retention", None)
    candidates = []
    for i in ids:
        proposed = source_candidates[i]
        model, data, _ = load_scene(base)
        right, left = [rebase_closed_wrist(model, data, h) for h in proposed["hands"]]
        for hand in (right, left):
            set_root_frame(
                model,
                data,
                hand["root_body"],
                hand["fixture_prefix"],
                hand["base_position_m"],
                hand["base_rotation"],
            )
        objq = model.joint("task_object_free").qposadr[0]
        data.qpos[objq : objq + 7] = old_plan["settling"]["object_qpos"]
        indices = [robot_indices(model, h["root_body"]) for h in (right, left)]
        combined = {
            key: np.concatenate([ids[key] for ids in indices])
            for key in ("joints", "q", "dofs", "geoms")
        }
        path_check = path_audit(
            model,
            data,
            combined,
            [
                np.concatenate([h[key] for h in (right, left)])
                for key in ("q_pregrasp", "q_open", "q_closed")
            ],
        )
        if (
            not path_check["within_joint_limits"]
            or path_check["max_penetration_m"] > base["planning"]["max_penetration_m"]
        ):
            raise ValueError("Rebased path failed collision or joint-limit admission")
        c = deepcopy(parent["candidate"])
        for key in (
            "root_body",
            "fixture_prefix",
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
            if key in right and key not in ("root_body", "fixture_prefix"):
                c[key] = right[key]
        c.update(
            candidate_id=i,
            support=left,
            accepted=True,
            rejected_for=[],
            bilateral_plan_sha256=config["plan_sha256"],
            score=proposed["paired_path_max_penetration_m"],
            rebased_path_check=path_check,
        )
        candidates.append(c)
    (out / "planning").mkdir()
    write_json(
        out / "planning/plan.json",
        dict(old_plan, candidates=candidates, accepted=len(candidates)),
    )
    base["planning_directory"] = str(out / "planning")
    write_json(out / "trial-config.json", base)
    rows = []
    for c in candidates:
        result = trial(base, c, out / f"candidate-{c['candidate_id']}")
        score = assess_motion_condition(
            result, motion_condition, config["fidelity_limits"]
        )
        row = dict(
            candidate_id=c["candidate_id"],
            success=result["success"],
            rejected_for=result["rejected_for"],
            hold_slip_m=result["hold_slip_m"],
            support_wrist_drift_m=result["support_wrist_drift_m"],
            hold_opposed_contact_fraction=result["hold_opposed_contact_fraction"],
            support_load_fraction=result["support_load_fraction"],
            **score,
        )
        rows.append(row)
        write_json(out / "progress.json", dict(trials=rows))
        print(row, flush=True)
    report = dict(
        schema="real2sim2real.bilateral-grasp-evaluation/1",
        config=config,
        trials=rows,
        motion_condition=motion_condition,
        baseline_replaced=False,
        arm_dynamics_validated=False,
        hardware_validated=False,
        implementation_sha256=digest(out / "implementation/bilateral_evaluate.py"),
    )
    write_json(out / "evaluation.json", report)
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
        raise ValueError("CPU-only evaluation required")
    run(json.loads(a.config.read_text()))


if __name__ == "__main__":
    main()
