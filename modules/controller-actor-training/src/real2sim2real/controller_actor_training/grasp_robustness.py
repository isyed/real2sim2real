"""Finite controller comparison on a frozen object grasp and separate validation."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .object_grasp_pilot import load_plan, trial, verify_selected_trial


def summary(result, directory):
    hold = [r for r in result["records"] if r["phase"] == "hold"]
    keys = [
        "candidate_id",
        "success",
        "rejected_for",
        "hold_minimum_clearance_m",
        "hold_opposed_contact_fraction",
        "hold_slip_m",
        "hold_relative_world_displacement_m",
        "hold_slip_frame",
        "max_robot_penetration_m",
        "max_robot_table_penetration_m",
        "states_sha256",
        "controller",
        "mass_scale",
        "friction",
    ]
    return dict(
        directory=str(directory),
        **{k: result[k] for k in keys},
        hold_max_rotation_deg=max(
            (r["rotation_change_deg"] for r in hold), default=None
        ),
        hold_mean_normal_force_by_group_n={
            group: float(
                np.mean([r["normal_contact_force_by_group_n"][group] for r in hold])
            )
            for group in ("thumb", "index", "middle", "ring", "pinky", "palm_or_other")
        }
        if hold
        else {},
    )


def compare(config):
    settings = config["robustness"]
    output = Path(settings["output"])
    parent = Path(settings["parent_evaluation"])
    if digest(parent) != settings["parent_sha256"]:
        raise ValueError("Parent evaluation changed")
    evaluation = json.loads(parent.read_text())
    selected = json.loads(
        (parent.parent / evaluation["selected_trial"] / "result.json").read_text()
    )
    verify_selected_trial(evaluation, selected, load_plan(config))
    if (
        digest(parent.parent / evaluation["selected_trial"] / "states.npz")
        != selected["states_sha256"]
    ):
        raise ValueError("Parent states changed")
    controllers, cases = settings["controllers"], settings["development_cases"]
    validation_cases = settings["validation_cases"]
    if (
        not controllers
        or not cases
        or len(controllers) * len(cases) > 24
        or not 1 <= len(validation_cases) <= 12
    ):
        raise ValueError("Use at most 24 development trials and 12 validation trials")
    if len({x["name"] for x in controllers}) != len(controllers):
        raise ValueError("Controller names must be unique")
    for row in [*controllers, *cases, *validation_cases]:
        if (
            not row["name"]
            or Path(row["name"]).name != row["name"]
            or row["name"] in (".", "..")
        ):
            raise ValueError("Use a local trial name")
    for controller in controllers:
        if set(controller) - {"name", "contact_feedback", "orientation_feedback"}:
            raise ValueError(
                "Only declared contact/orientation controller corrections may vary"
            )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", config)
    rows, scores = [], []
    base = deepcopy(config)
    base["rollout"]["close_offset_rad"] = selected["controller"]["close_offset_rad"]
    for controller in controllers:
        current = deepcopy(base)
        for name in ("contact_feedback", "orientation_feedback"):
            current[name] = controller.get(name)
        group = []
        for case in cases:
            name = controller["name"] + "--" + case["name"]
            result = trial(
                current,
                selected["candidate"],
                output / name,
                mass_scale=case["mass_scale"],
                friction=case["friction"],
            )
            row = summary(result, name)
            rows.append(row)
            group.append(row)
            write_json(output / "progress.json", dict(trials=rows))
            print(name, result["success"], result["rejected_for"], flush=True)
        scores.append(
            dict(
                name=controller["name"],
                passed=sum(r["success"] for r in group),
                worst_clearance_m=min(r["hold_minimum_clearance_m"] for r in group),
                worst_contact_fraction=min(
                    r["hold_opposed_contact_fraction"] for r in group
                ),
            )
        )
    eligible = [r for r in scores if r["passed"] == len(cases)]
    chosen = (
        max(
            eligible,
            key=lambda r: (r["worst_contact_fraction"], r["worst_clearance_m"]),
        )
        if eligible
        else None
    )
    selection = dict(
        selected_controller=None if chosen is None else chosen["name"],
        development_scores=scores,
        parent_evaluation_sha256=digest(parent),
        thresholds_relaxed=False,
    )
    write_json(output / "selection.json", selection)
    validation = []
    if chosen:
        controller = next(c for c in controllers if c["name"] == chosen["name"])
        current = deepcopy(base)
        current["rollout"]["hold_s"] = settings["validation_hold_s"]
        for name in ("contact_feedback", "orientation_feedback"):
            current[name] = controller.get(name)
        for case in validation_cases:
            name = "validation--" + case["name"]
            result = trial(
                current,
                selected["candidate"],
                output / name,
                mass_scale=case["mass_scale"],
                friction=case["friction"],
            )
            validation.append(summary(result, name))
            print(name, result["success"], result["rejected_for"], flush=True)
    decision = (
        "passes_declared_controller_cases"
        if validation and all(r["success"] for r in validation)
        else "development_pass_validation_failed"
        if chosen
        else "no_robust_controller_in_bounded_search"
    )
    report = dict(
        schema="real2sim2real.grasp-robustness/1",
        decision=decision,
        selection=selection,
        development=rows,
        validation=validation,
        human_reference_changed=False,
        geometry_changed=False,
        policy_trained=False,
        arm_reachability_validated=False,
    )
    write_json(output / "comparison.json", report)
    return report


def validation_decision(development_passed, rows):
    if not development_passed:
        return "diagnostic_only_development_failed"
    return (
        "passes_declared_controller_cases"
        if rows and all(r["success"] for r in rows)
        else "validation_failed"
    )


def validate_fixed(config):
    """Longer diagnostics never erase a failed development admission."""
    validation = config["robustness_validation"]
    parent, protocol = Path(validation["comparison"]), Path(validation["protocol"])
    if (
        digest(parent) != validation["comparison_sha256"]
        or digest(protocol) != validation["protocol_sha256"]
    ):
        raise ValueError("Frozen development comparison/protocol changed")
    report, source = json.loads(parent.read_text()), json.loads(protocol.read_text())
    name = validation["controller_name"]
    controllers = [c for c in source["robustness"]["controllers"] if c["name"] == name]
    if len(controllers) != 1:
        raise ValueError("A unique tested controller is required")
    original = Path(source["robustness"]["parent_evaluation"])
    if digest(original) != source["robustness"]["parent_sha256"]:
        raise ValueError("Parent grasp evaluation changed")
    evaluation = json.loads(original.read_text())
    selected = json.loads(
        (original.parent / evaluation["selected_trial"] / "result.json").read_text()
    )
    verify_selected_trial(evaluation, selected, load_plan(source))
    development_passed = any(
        s["name"] == name
        and s["passed"] == len(source["robustness"]["development_cases"])
        for s in report["selection"]["development_scores"]
    )
    current = deepcopy(source)
    current["rollout"]["close_offset_rad"] = selected["controller"]["close_offset_rad"]
    current["rollout"]["hold_s"] = validation["hold_s"]
    for key in ("contact_feedback", "orientation_feedback"):
        current[key] = controllers[0].get(key)
    cases = validation["cases"]
    if not 1 <= len(cases) <= 12:
        raise ValueError("Use at most twelve fixed validation cases")
    output = Path(validation["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", config)
    rows = []
    for case in cases:
        case_name = case["name"]
        if (
            not case_name
            or Path(case_name).name != case_name
            or case_name in (".", "..")
        ):
            raise ValueError("Use local case names")
        result = trial(
            current,
            selected["candidate"],
            output / case_name,
            mass_scale=case["mass_scale"],
            friction=case["friction"],
        )
        rows.append(summary(result, case_name))
        print(case_name, result["success"], result["rejected_for"], flush=True)
    final = dict(
        schema="real2sim2real.grasp-robustness-validation/1",
        parent_comparison_sha256=digest(parent),
        controller_name=name,
        development_passed=development_passed,
        trials=rows,
        decision=validation_decision(development_passed, rows),
        thresholds_relaxed=False,
        policy_trained=False,
    )
    write_json(output / "validation.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run

        command = [
            sys.executable,
            "-m",
            __spec__.name,
            "--config",
            str(args.config.resolve()),
            "--worker",
        ]
        if args.validate:
            command.append("--validate")
        return run(
            command,
            timeout=config["robustness_validation" if args.validate else "robustness"][
                "deadline_s"
            ],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide the GPU for this CPU experiment")
    return validate_fixed(config) if args.validate else compare(config)


if __name__ == "__main__":
    main()
