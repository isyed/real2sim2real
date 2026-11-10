"""Matched, finite retention comparison with a frozen reconstructed trajectory."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .guided_motion import load_reference
from .guided_motion_pilot import fidelity
from .grasp_robustness import summary
from .object_grasp_pilot import trial


def assess(result, limits):
    ref = load_reference(result["configuration"]["reference_guidance"])
    if result["reference_start_time_s"] is None:
        return dict(combined_pass=False, reference_pass=False, fidelity=None)
    score = fidelity(
        result["records"],
        ref,
        result["reference_start_time_s"],
        np.array(result["reference_anchor_world_from_object"]),
    )
    complete = (
        result["simulation_final_time_s"]
        >= result["simulation_requested_duration_s"] - 1e-6
    )
    passed = (
        complete
        and score["samples"] > 0
        and score["maximum_position_error_m"] <= limits["position_m"]
        and score["maximum_rotation_error_deg"] <= limits["rotation_deg"]
    )
    return dict(
        fidelity=score,
        reference_pass=passed,
        combined_pass=passed and result["success"],
        complete=complete,
    )


def run(config):
    for p, sha in config["sources"].items():
        if digest(Path(p)) != sha:
            raise ValueError("Pinned retention input changed")
    parent = json.loads(Path(config["parent_trial"]).read_text())
    if (
        digest(Path(config["parent_trial"]).with_name("states.npz"))
        != parent["states_sha256"]
    ):
        raise ValueError("Parent state archive changed")
    source_evaluation = json.loads(Path(config["parent_evaluation"]).read_text())
    limits = source_evaluation["config"]["fidelity_limits"]
    base = parent["configuration"]
    if base.get("grasp_retention"):
        raise ValueError("Require an unchanged uncorrected baseline")
    controllers, validation = config["controllers"], config["validation"]
    if not 1 <= len(controllers) <= 3 or not 1 <= len(validation) <= 4:
        raise ValueError("Bounded controller/validation list required")
    names = [c["name"] for c in controllers] + [c["name"] for c in validation]
    if len(set(names)) != len(names) or any(
        Path(n).name != n or n in ("", ".", "..", "baseline") for n in names
    ):
        raise ValueError("Distinct local experiment names required")
    for v in validation:
        if v["hold_s"] < base["rollout"]["hold_s"]:
            raise ValueError("Validation may not shorten hold")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        dict(
            config=config,
            fidelity_limits=limits,
            physical_gates_unchanged=True,
            capture_motion_unchanged=True,
            development_selection="lowest primary slip among complete physical-and-reference passes",
            validation_selection_frozen=True,
        ),
    )
    rows = []

    def execute(name, control, case=None):
        settings = deepcopy(base)
        if control is not None:
            settings["grasp_retention"] = control
        kwargs = {}
        if case is not None:
            settings["rollout"]["hold_s"] = case["hold_s"]
            kwargs = dict(mass_scale=case["mass_scale"], friction=case["friction"])
        result = trial(settings, parent["candidate"], out / name, **kwargs)
        row = dict(**summary(result, name), **assess(result, limits))
        if name == "baseline":
            row["baseline_states_identical"] = (
                result["states_sha256"] == parent["states_sha256"]
            )
            if not row["baseline_states_identical"]:
                raise ValueError("Unchanged guided baseline regression")
        rows.append(row)
        write_json(out / "progress.json", dict(trials=rows))
        print(
            name,
            row["combined_pass"],
            result["rejected_for"],
            "slip",
            row["hold_slip_m"],
            flush=True,
        )
        return row

    execute("baseline", None)
    development = [execute(c["name"], c["settings"]) for c in controllers]
    admitted = [r for r in development if r["combined_pass"]]
    best = min(admitted, key=lambda r: r["hold_slip_m"]) if admitted else None
    selection = dict(
        selected=None if best is None else best["directory"],
        development_pass=best is not None,
        validation=[],
    )
    write_json(out / "selection.json", selection)
    if best:
        chosen = next(
            c["settings"] for c in controllers if c["name"] == best["directory"]
        )
        for case in validation:
            execute("validation-baseline--" + case["name"], None, case)
            selection["validation"].append(
                execute("validation-selected--" + case["name"], chosen, case)
            )
    report = dict(
        schema="real2sim2real.retention-experiment/1",
        config=config,
        trials=rows,
        selection=selection,
        decision="passes_declared_retention_cases"
        if best and all(r["combined_pass"] for r in selection["validation"])
        else "nominal_pass_validation_failed"
        if best
        else "no_retention_candidate_admitted",
        control_ready=False,
        human_finger_reference_used=False,
        policy_trained=False,
        implementation_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("grasp_retention.py"),
                Path(__file__).with_name("object_grasp_pilot.py"),
            ]
        },
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU experiment requires CUDA hidden")
    run(c)


if __name__ == "__main__":
    main()
