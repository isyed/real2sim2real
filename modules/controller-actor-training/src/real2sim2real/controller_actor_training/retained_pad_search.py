"""Bounded pad replanning around a retained bimanual grasp and motion target."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from real2sim2real.artifacts import digest, write_json


def run(config):
    import mujoco
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_wrist_frame,
        set_root_frame,
        trial,
    )
    from .pad_grasp_planning import fit_candidate, path_audit
    from .grasp_posture_search import rotation_margin, local_name
    from .grasp_robustness import summary
    from .retention_experiment import assess
    from real2sim2real.retargeting.contact_layout import contact_targets
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    for p, sha in config["sources"].items():
        if digest(Path(p)) != sha:
            raise ValueError("Pinned pad-search input changed")
    parent = json.loads(Path(config["parent_trial"]).read_text())
    if (
        not parent["success"]
        or digest(Path(config["parent_trial"]).with_name("states.npz"))
        != parent["states_sha256"]
    ):
        raise ValueError("Require unchanged successful parent and states")
    base = parent["configuration"]
    source_plan = load_plan(base)
    if parent["candidate"] not in source_plan["candidates"]:
        raise ValueError("Parent is absent from the retained plan")
    variants = config["variants"]
    if not 1 <= len(variants) <= 4:
        raise ValueError("One to four predeclared pad proposals required")
    names = [local_name(v["name"]) for v in variants]
    if len(set(names)) != len(names) or "baseline" in names:
        raise ValueError("Distinct proposal names required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    candidates = [deepcopy(parent["candidate"])]
    for number, variant in enumerate(variants):
        model, data, mesh = load_scene(base)
        candidate = deepcopy(parent["candidate"])
        set_wrist_frame(
            model, data, base, candidate["base_position_m"], candidate["base_rotation"]
        )
        s = candidate["support"]
        set_root_frame(
            model,
            data,
            s["root_body"],
            s["fixture_prefix"],
            s["base_position_m"],
            s["base_rotation"],
        )
        mujoco.mj_resetData(model, data)
        idx = robot_indices(model, base["robot"]["root_body"])
        support = robot_indices(model, s["root_body"])
        data.qpos[support["q"]] = s["q_pregrasp"]
        qid = model.joint("task_object_free").qposadr[0]
        data.qpos[qid : qid + 7] = source_plan["settling"]["object_qpos"]
        world = mesh.copy()
        world.apply_transform(source_plan["settling"]["settled_world_from_object"])
        targets = contact_targets(
            world,
            candidate["pair"],
            variant["finger_offsets_m"],
            direction=candidate["thumb_side_direction"],
        )
        candidate.update(
            candidate_id=2000 + number,
            proposal_name=variant["name"],
            accepted=False,
            rejected_for=[],
        )
        if targets is None:
            candidate["rejected_for"] = ["missing_contact_surface"]
        else:
            settings = deepcopy(base)
            settings["pad_planning"] = config["pad_planning"]
            fit, opened, _, extra, targets = fit_candidate(
                model,
                data,
                settings,
                world,
                targets,
                np.array(candidate["q_grasp"]),
                idx,
                candidate["joint_names"].index("fixture_z"),
            )
            pre = opened.copy()
            xyz = [candidate["joint_names"].index("fixture_" + a) for a in "xyz"]
            pre[xyz] += candidate["approach_offset_m"]
            pad = extra["pad_plan"]
            pad["vertical_path_diagnostic"] = pad["path"]
            pad["path"] = path_audit(model, data, idx, [pre, opened, extra["q_closed"]])
            reasons = [
                r
                for r in pad["rejected_for"]
                if r not in ("pad_closing_path_penetration", "pad_path_joint_limits")
            ]
            if pad["path"]["max_penetration_m"] > base["planning"]["max_penetration_m"]:
                reasons.append("pad_closing_path_penetration")
            if not pad["path"]["within_joint_limits"]:
                reasons.append("pad_path_joint_limits")
            if max(fit["site_errors_m"]) > base["planning"]["contact_error_m"]:
                reasons.append("contact_target_error")
            pad["rejected_for"] = reasons
            data.qpos[idx["q"]] = fit["q"]
            candidate.update(
                **extra,
                q_grasp=fit["q"].tolist(),
                q_open=opened.tolist(),
                q_pregrasp=pre.tolist(),
                targets_m=targets.tolist(),
                site_errors_m=fit["site_errors_m"].tolist(),
                accepted=not reasons,
                rejected_for=reasons,
                planned_wrist_margin_rad=rotation_margin(
                    model, data.qpos[None], "fixture_"
                ),
            )
        candidates.append(candidate)
        print(variant["name"], candidate["rejected_for"], flush=True)
        write_json(out / "planning-progress.json", dict(candidates=candidates))
    plan = dict(source_plan, candidates=candidates)
    (out / "planning").mkdir()
    write_json(out / "planning/plan.json", plan)
    rows = []

    def execute(name, candidate, case=None):
        c = deepcopy(base)
        c["planning_directory"] = str(out / "planning")
        if candidate.get("pad_plan"):
            c["rollout"]["close_offset_rad"] = 0
        kwargs = {}
        if case:
            c["rollout"]["hold_s"] = case["hold_s"]
            kwargs = dict(mass_scale=case["mass_scale"], friction=case["friction"])
        r = trial(c, candidate, out / name, **kwargs)
        row = dict(**summary(r, name), **assess(r, config["fidelity_limits"]))
        if name == "baseline" and r["states_sha256"] != parent["states_sha256"]:
            raise ValueError("Baseline regression")
        rows.append(row)
        write_json(out / "progress.json", dict(trials=rows))
        print(name, row["combined_pass"], r["rejected_for"], flush=True)
        return row

    execute("baseline", candidates[0])
    passing = []
    for c in candidates[1:]:
        if c["accepted"]:
            r = execute(c["proposal_name"], c)
            if r["combined_pass"]:
                passing.append((r, c))
    selected = min(passing, key=lambda x: x[0]["hold_slip_m"])[1] if passing else None
    write_json(
        out / "selection.json",
        dict(
            selected=None if selected is None else selected["proposal_name"],
            frozen_before_validation=True,
        ),
    )
    if selected:
        for case in config["validation"]:
            if case["hold_s"] < base["rollout"]["hold_s"]:
                raise ValueError("Cannot shorten validation hold")
            execute(
                "validation-baseline--" + local_name(case["name"]), candidates[0], case
            )
            execute("validation-selected--" + local_name(case["name"]), selected, case)
    report = dict(
        schema="real2sim2real.retained-pad-search/1",
        config=config,
        trials=rows,
        planned_candidates=[
            dict(
                name=c["proposal_name"],
                accepted=c["accepted"],
                rejected_for=c["rejected_for"],
                pad_plan=c.get("pad_plan"),
                site_errors_m=c.get("site_errors_m"),
            )
            for c in candidates[1:]
        ],
        selected=None if selected is None else selected["proposal_name"],
        control_ready=False,
        thresholds_relaxed=False,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "comparison.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
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
            timeout=c["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    run(c)


if __name__ == "__main__":
    main()
