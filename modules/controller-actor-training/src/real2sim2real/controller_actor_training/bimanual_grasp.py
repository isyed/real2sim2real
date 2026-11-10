"""Bounded CPU second-hand support experiment on a preserved single-hand baseline."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def evaluate(config):
    from .object_grasp_pilot import load_plan, trial

    output = Path(config["output"]) / config.get("evaluation_directory", "evaluation")
    output.mkdir(exist_ok=False)
    plan = load_plan(config)
    settings = deepcopy(config)
    parent = json.loads(Path(config["parent"]["trial"]).read_text())
    if digest(Path(config["parent"]["trial"])) != config["parent"]["trial_sha256"]:
        raise ValueError("Baseline changed")
    settings["rollout"]["close_offset_rad"] = parent["controller"]["close_offset_rad"]
    rows = []
    for candidate in sorted(
        [r for r in plan["candidates"] if r["accepted"]], key=lambda r: r["score"]
    )[:6]:
        name = f"support-{candidate['candidate_id']}"
        result = trial(settings, candidate, output / name)
        rows.append(
            dict(
                directory=name,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("records", "candidate", "configuration")
                },
            )
        )
        print(f"{name}: {result['success']} {result['rejected_for']}", flush=True)
    passed = [r for r in rows if r["success"]]
    selected = min(passed, key=lambda r: r["hold_slip_m"]) if passed else None
    report = dict(
        schema="real2sim2real.bimanual-grasp-evaluation/1",
        trials=rows,
        variants=[],
        selected_trial=selected["directory"] if selected else None,
        selected_candidate=selected["candidate_id"] if selected else None,
        decision="nominal_bimanual_support_only"
        if selected
        else "no_validated_bimanual_support",
        object_assistance=False,
        full_episode_control_validated=False,
        plan_sha256=digest(
            Path(config["output"])
            / config.get("planning_directory", "planning")
            / "plan.json"
        ),
    )
    write_json(output / "evaluation.json", report)
    return report


def ablate_support(config):
    """Park the second hand for matched-duration diagnostic comparisons only."""
    from .object_grasp_pilot import (
        load_plan,
        trial,
        verify_selected_trial,
        verify_frozen_controller,
    )

    settings = config["validation"]
    parent = Path(settings["parent_evaluation"])
    if digest(parent) != settings["parent_sha256"]:
        raise ValueError("Pinned selection changed")
    evaluation = json.loads(parent.read_text())
    if evaluation["selected_trial"] is None:
        raise ValueError("Select a nominally successful bimanual grasp first")
    directory = parent.parent / evaluation["selected_trial"]
    selected = json.loads((directory / "result.json").read_text())
    if digest(directory / "states.npz") != selected["states_sha256"]:
        raise ValueError("Selected states changed")
    verify_selected_trial(evaluation, selected, load_plan(config))
    verify_frozen_controller(config, selected)
    cases = config["ablation"]["cases"]
    if not 1 <= len(cases) <= 4:
        raise ValueError("Bounded ablation case list required")
    run_config = deepcopy(config)
    run_config["support_controller"]["enabled"] = False
    run_config["rollout"]["close_offset_rad"] = selected["controller"][
        "close_offset_rad"
    ]
    output = Path(config["output"]) / config["ablation"]["directory"]
    output.mkdir(exist_ok=False)
    rows = []
    for case in cases:
        name = case["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Use a local case name")
        result = trial(
            run_config,
            selected["candidate"],
            output / name,
            mass_scale=case["mass_scale"],
            friction=case["friction"],
        )
        rows.append(
            dict(
                directory=name,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("records", "candidate", "configuration")
                },
            )
        )
        print(f"Parked support {name}: {result['rejected_for']}", flush=True)
    report = dict(
        schema="real2sim2real.support-ablation/1",
        trials=rows,
        source_evaluation_sha256=digest(parent),
        bimanual_promotion_allowed=False,
        note="Same primary controller, scene and scoring window; support hand remains parked. Support gates necessarily fail.",
    )
    write_json(output / "ablation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["prepare", "plan", "evaluate", "validate", "ablate", "review"],
        required=True,
    )
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
            timeout=config["deadlines_s"][args.phase],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide GPU for CPU support experiments")
    if args.phase == "prepare":
        from .multi_hand_scene import append_hand

        return append_hand(config)
    if args.phase == "plan":
        from .support_planning import plan_support

        return plan_support(config)
    if args.phase == "evaluate":
        return evaluate(config)
    if args.phase == "ablate":
        return ablate_support(config)
    if args.phase == "validate":
        from .object_grasp_pilot import validate_selected

        return validate_selected(config)
    from .object_grasp_review import review

    return review(config)


if __name__ == "__main__":
    main()
