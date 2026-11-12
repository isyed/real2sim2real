"""Export only named strategy development runs, preserving failed geometry gates."""

import argparse
import json
from pathlib import Path
from copy import deepcopy

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json


def _validate_outcomes(evaluation, plan):
    if (
        evaluation["geometric_pass"] != plan["accepted"]
        or evaluation["rejected_for"] != plan["rejected_for"]
    ):
        raise ValueError("Planning and evaluation verdicts disagree")
    if evaluation["hold"]["executed"] and not plan["accepted"]:
        raise ValueError("Physics cannot bypass rejected geometry")


def load_run(run):
    """Read matching pinned artifacts before exporting feedback or review media."""
    evaluation, plan = pinned_json(run["evaluation"]), pinned_json(run["plan"])
    if Path(run["plan"]["path"]).resolve() != Path(
        run["evaluation"]["path"]
    ).resolve().with_name("plan.json"):
        raise ValueError("Plan and evaluation must belong to one archived run")
    _validate_outcomes(evaluation, plan)
    return evaluation, plan


def summarize(evaluation, plan):
    _validate_outcomes(evaluation, plan)
    return {
        "parameters": deepcopy(plan["parameters"]),
        "geometric_pass": plan["accepted"],
        "rejected_for": list(plan["rejected_for"]),
        "hands": [
            {
                key: deepcopy(hand.get(key))
                for key in (
                    "root_body",
                    "targets_m",
                    "outward_normals",
                    "site_errors_m",
                    "maximum_target_error_m",
                    "maximum_normal_error_deg",
                    "patches",
                    "path",
                    "nfev",
                    "rejected_for",
                )
            }
            for hand in plan["hands"]
        ],
        "paired_path": deepcopy(plan["paired_path"]),
        "paired_gravity_wrench": deepcopy(plan["gravity_wrench"]),
        "hold": deepcopy(evaluation["hold"]),
        "interpretation": [
            "Errors and penetration are sampled kinematic model checks, not physical measurements.",
            "Pad patch counts refer to the reconstructed surface near each placed collision pad, not independent observations.",
            "Missing contact samples cause the wrench screen to reject without establishing an infeasible complete force solution.",
            "An unexecuted hold supplies no measured forces, slip or task success rate.",
            "The tool graph can complete while geometry and task acceptance remain rejected.",
        ],
    }


def build(config):
    if not 1 <= len(config["runs"]) <= 8:
        raise ValueError("Explicit bounded development run set required")
    packets = []
    for run in config["runs"]:
        evaluation, plan = load_run(run)
        packets.append({"label": run["label"], **summarize(evaluation, plan)})
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "grasp-strategy-feedback/1",
        "runs": packets,
        "heldout_results_included": False,
    }
    write_json(output / "feedback.json", report)
    write_json(
        output / "provenance.json",
        {"config": config, "implementation_sha256": digest(Path(__file__))},
    )
    for run in config["runs"]:
        load_run(run)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()))
