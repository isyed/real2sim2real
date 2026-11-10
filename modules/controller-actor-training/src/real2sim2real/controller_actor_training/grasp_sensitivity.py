"""Frozen-controller payload/friction trials from a complete admitted grasp.

Each case starts from acquisition. Only payload and sliding friction vary; the
source candidate, reference clock, controller, duration and acceptance stay fixed.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def validate_cases(cases):
    if not 2 <= len(cases) <= 12:
        raise ValueError("Two to twelve cases including a nominal replay required")
    names = set()
    for case in cases:
        if set(case) != {"name", "mass_scale", "sliding_friction"}:
            raise ValueError("Only explicit payload/friction perturbations allowed")
        name = case["name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
            or name in names
        ):
            raise ValueError("Distinct local case names required")
        names.add(name)
        mass, friction = case["mass_scale"], case["sliding_friction"]
        if (
            isinstance(mass, bool)
            or not math.isfinite(mass)
            or not 0.5 <= mass <= 2
            or (
                friction is not None
                and (
                    isinstance(friction, bool)
                    or not math.isfinite(friction)
                    or not 0.2 <= friction <= 1.2
                )
            )
        ):
            raise ValueError("Case exceeds finite payload/friction bounds")
    if cases[0] != dict(name="nominal", mass_scale=1.0, sliding_friction=None):
        raise ValueError("First case must be the unchanged nominal replay")


def load_frozen_trial(config):
    if set(config) != {
        "source_result",
        "source_sha256",
        "protocol_sha256",
        "output",
        "deadline_s",
        "cases",
    }:
        raise ValueError(
            "Frozen sensitivity cannot accept controller or gate overrides"
        )
    validate_cases(config["cases"])
    if not math.isfinite(config["deadline_s"]) or not 0 < config["deadline_s"] <= 1800:
        raise ValueError("Finite stage deadline of at most 1800 seconds required")
    path = Path(config["source_result"])
    protocol_path = path.parent.parent / "protocol.json"
    if (
        digest(path) != config["source_sha256"]
        or digest(protocol_path) != config["protocol_sha256"]
    ):
        raise ValueError("Pinned source result/protocol changed")
    source = json.loads(path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (
        source["success"] is not True
        or not math.isfinite(source["simulation_final_time_s"])
        or not math.isfinite(source["simulation_requested_duration_s"])
        or source["simulation_requested_duration_s"] <= 0
        or source["simulation_final_time_s"]
        < source["simulation_requested_duration_s"] - 1e-6
        or digest(path.with_name("states.npz")) != source["states_sha256"]
        or source["mass_scale"] != 1
        or source["friction"] != source["configuration"]["physics"]["friction"]
    ):
        raise ValueError("Unchanged complete nominal passing grasp and states required")
    limits = protocol["fidelity_limits"]
    if source["configuration"].get("reference_guidance"):
        from .retention_experiment import assess

        if (
            protocol.get("motion_condition") != "demonstrated"
            or not assess(source, limits)["combined_pass"]
        ):
            raise ValueError("Source must pass physical and reference acceptance")
    elif protocol.get("motion_condition") != "stationary_diagnostic":
        raise ValueError("Explicit stationary or demonstrated source required")
    return source, limits


def run(config):
    from .object_grasp_pilot import trial
    from .grasp_robustness import summary
    from .bilateral_evaluate import assess_motion_condition

    source, limits = load_frozen_trial(config)
    base = source["configuration"]
    condition = (
        "demonstrated" if base.get("reference_guidance") else "stationary_diagnostic"
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        dict(
            config,
            fidelity_limits=limits,
            motion_condition=condition,
            controller_frozen=True,
        ),
    )
    implementation = {
        filename: digest(Path(__file__).with_name(filename))
        for filename in (
            "grasp_sensitivity.py",
            "object_grasp_pilot.py",
            "contact_force_control.py",
            "payload_pressure.py",
            "support_control.py",
            "guided_motion.py",
            "wrist_coordination.py",
            "grasp_hold.py",
            "grasp_feedback.py",
            "retention_experiment.py",
        )
    }
    (out / "implementation").mkdir()
    for filename, sha in implementation.items():
        snapshot = out / "implementation" / filename
        snapshot.write_bytes(Path(__file__).with_name(filename).read_bytes())
        if digest(snapshot) != sha:
            raise ValueError("Implementation changed while preserving its snapshot")
    write_json(out / "implementation.json", implementation)
    rows = []
    nominal_identical = False
    for case in config["cases"]:
        result = trial(
            deepcopy(base),
            source["candidate"],
            out / case["name"],
            mass_scale=case["mass_scale"],
            friction=case["sliding_friction"],
        )
        score = assess_motion_condition(result, condition, limits)
        if case["name"] == "nominal":
            nominal_identical = result["states_sha256"] == source["states_sha256"]
        row = dict(
            **summary(result, out / case["name"]),
            **score,
            result_sha256=digest(out / case["name"] / "result.json"),
        )
        rows.append(row)
        write_json(out / "progress.json", dict(trials=rows))
        print(case["name"], result["success"], score, flush=True)
        if case["name"] == "nominal" and not nominal_identical:
            break
    passed = (
        nominal_identical
        and len(rows) == len(config["cases"])
        and all(
            r["complete"]
            and (
                r["combined_pass"]
                if condition == "demonstrated"
                else r["diagnostic_hold_pass"]
            )
            for r in rows
        )
    )
    report = dict(
        schema="real2sim2real.grasp-sensitivity/1",
        config=config,
        source_states_sha256=source["states_sha256"],
        trials=rows,
        nominal_states_identical=nominal_identical,
        all_declared_cases_pass=passed,
        motion_condition=condition,
        fidelity_limits=limits,
        controller_retuned=False,
        reference_changed=False,
        thresholds_relaxed=False,
        arm_dynamics_validated=False,
        hardware_validated=False,
        implementation_sha256=implementation,
    )
    write_json(out / "sensitivity.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run as bounded

        # Validate before starting a bounded subprocess or creating any outputs.
        load_frozen_trial(config)
        return bounded(
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only sensitivity evaluation required")
    run(config)


if __name__ == "__main__":
    main()
