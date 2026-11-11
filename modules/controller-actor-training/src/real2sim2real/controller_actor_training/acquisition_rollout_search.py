"""Finite acquisition search using complete free-object stationary rollouts.

Ranks development candidates without replacing their original task verdicts.
Robot acquisition schedules vary; physical parameters, force/torque budgets,
reference clock, acquisition anchor and acceptance limits remain frozen.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .acquisition_profile import identity, validate
from .continuous_retention_audit import load_readiness


def read(spec):
    if set(spec) != {"path", "sha256"}:
        raise ValueError("Pinned artifact path and sha256 required")
    path = Path(spec["path"]).resolve()
    if digest(path) != spec["sha256"]:
        raise ValueError("Pinned rollout-search input changed")
    return path, json.loads(path.read_text())


def validate_names(profiles):
    names = [p["name"] for p in profiles]
    reserved = {"implementation", "protocol", "progress", "search", "repeat-selection"}
    if (
        len(set(names)) != len(names)
        or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", n) or n in reserved
            for n in names
        )
        or any(n + "-repeat" in names for n in names)
    ):
        raise ValueError(
            "Distinct safe candidate names must not overwrite search/repeat artifacts"
        )


def state_archive(path, report):
    for filename, key in (
        ("scene.xml", "scene_sha256"),
        ("states.npz", "states_sha256"),
    ):
        if digest(path.with_name(filename)) != report[key]:
            raise ValueError("Saved physics evidence changed")
    with np.load(path.with_name("states.npz"), allow_pickle=False) as archive:
        if any(not np.isfinite(archive[k]).all() for k in archive.files):
            raise ValueError("Nonfinite saved physics states")
        return {k: archive[k] for k in archive.files}


def fixed_limits(baseline):
    audit_path, audit = read(
        dict(
            path=baseline["config"]["audit"], sha256=baseline["config"]["audit_sha256"]
        )
    )
    for name, sha in audit["sources"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Search source geometry/commands changed")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    task = json.loads(Path(parent["configuration"]["task"]).read_text())
    return dict(
        primary_drift_m=0.01,
        support_drift_m=0.01,
        minimum_clearance_m=task["goal"]["minimum_clearance_m"],
        maximum_robot_penetration_m=parent["configuration"]["planning"][
            "max_penetration_m"
        ],
        maximum_arm_penetration_m=audit["config"]["criteria"]["arm_penetration_m"],
        minimum_joint_margin_rad=audit["config"]["criteria"]["joint_margin_rad"],
        maximum_arm_velocity_ratio=1.0,
        maximum_reference_position_m=0.02,
        maximum_reference_rotation_deg=10.0,
        maximum_hold_rotation_deg=30.0,
        source_audit=str(audit_path),
        source_audit_sha256=digest(audit_path),
    )


def evaluate(report, baseline, actual_times, expected_times, limits):
    """Never score a shortened run or change its original acceptance decision."""
    if (
        report.get("schema") != "real2sim2real.arm-replay-pilot/1"
        or report.get("physics_run") is not True
        or report.get("motion_condition") != "stationary_hold"
        or report.get("object_state_written_only_at_initialization") is not True
    ):
        raise ValueError("Complete unassisted stationary arm physics required")
    excluded = {"output", "purpose", "deadline_s", "acquisition_profile"}
    strip = lambda cfg: {k: v for k, v in cfg.items() if k not in excluded}
    if strip(report["config"]) != strip(baseline["config"]):
        raise ValueError("Search may vary acquisition schedules only")
    if (
        report["scene_sha256"] != baseline["scene_sha256"]
        or report["physics_timestep_s"] != baseline["physics_timestep_s"]
        or report["controller_period_s"] != baseline["controller_period_s"]
    ):
        raise ValueError("Search physics or clocks changed")
    keys = (
        "frame_ids",
        "capture_duration_s",
        "capture_clock_origin_s",
        "acquisition_end_s",
        "reference_start_s",
    )
    if any(
        report["continuous_protocol"][k] != baseline["continuous_protocol"][k]
        for k in keys
    ):
        raise ValueError("Search reference clock or acquisition anchor changed")
    rows, original = report["records"], baseline["records"]
    complete = (
        np.array_equal(actual_times, expected_times)
        and len(rows) == len(original)
        and all(
            a["phase"] == b["phase"] and a["time_s"] == b["time_s"]
            for a, b in zip(rows, original)
        )
    )
    if not complete or "incomplete_pilot" in report["rejected_for"]:
        return dict(
            complete=False,
            eligible=False,
            passed=False,
            objective=None,
            rejection="Full frozen acquisition/hold clock was not completed",
        )
    if (
        report["continuous_metrics"]["retention_anchor_time_s"]
        != baseline["continuous_metrics"]["retention_anchor_time_s"]
    ):
        raise ValueError("Retention was reanchored")
    scalar_keys = (
        "hold_slip_m",
        "support_wrist_drift_m",
        "hold_minimum_clearance_m",
        "hold_max_rotation_deg",
        "max_robot_penetration_m",
        "max_arm_penetration_m",
        "minimum_dynamic_joint_margin_rad",
        "maximum_dynamic_velocity_ratio",
    )
    if any(report[k] is None or not np.isfinite(report[k]) for k in scalar_keys):
        raise ValueError("Finite complete-hold metrics required")
    hold = [r for r in rows if r["phase"] == "hold"]
    if not hold:
        raise ValueError("Frozen suspended hold cohort required")
    readiness = load_readiness(hold)
    ratios = dict(
        primary_drift=report["hold_slip_m"] / limits["primary_drift_m"],
        support_drift=report["support_wrist_drift_m"] / limits["support_drift_m"],
        clearance=limits["minimum_clearance_m"]
        / max(report["hold_minimum_clearance_m"], 1e-6),
        reference_position=max(r["reference_position_error_m"] for r in rows)
        / limits["maximum_reference_position_m"],
        reference_rotation=max(r["reference_rotation_error_deg"] for r in rows)
        / limits["maximum_reference_rotation_deg"],
        hold_rotation=report["hold_max_rotation_deg"]
        / limits["maximum_hold_rotation_deg"],
    )
    unsafe = (
        report["max_robot_penetration_m"] > limits["maximum_robot_penetration_m"]
        or report["max_arm_penetration_m"] > limits["maximum_arm_penetration_m"]
        or report["minimum_dynamic_joint_margin_rad"]
        < limits["minimum_joint_margin_rad"]
        or report["maximum_dynamic_velocity_ratio"]
        > limits["maximum_arm_velocity_ratio"]
    )
    unsafe = unsafe or "object_table_penetration" in report["rejected_for"]
    violations = sum(max(0.0, value - 1) for value in ratios.values())
    sustained = readiness["sampled_load_fraction"]["bilateral"]
    objective = violations + 0.25 * (1 - sustained) + 0.05 * ratios["hold_rotation"]
    passed = report["success"] is True and not report["rejected_for"]
    if passed and (unsafe or max(ratios.values()) > 1):
        raise ValueError("Task success contradicts frozen task limits")
    return dict(
        complete=True,
        eligible=True,
        passed=passed,
        unsafe=bool(unsafe),
        objective=float(objective),
        normalized_metrics=ratios,
        maximum_normalized_metric=max(ratios.values()),
        sustained_hold_load_fraction=sustained,
        hold_load_readiness=readiness,
        acceptance_changed=False,
    )


def rank(row):
    score = row.get("evaluation", {})
    if not score.get("eligible"):
        return (2, True, float("inf"), float("inf"), row["name"])
    return (
        not score["passed"],
        score["unsafe"],
        score["objective"],
        score["maximum_normalized_metric"],
        row["name"],
    )


def run(config):
    from real2sim2real.bounded_process import run as bounded

    required = {
        "seed_config",
        "baseline_result",
        "identity_result",
        "profiles",
        "parallel_trials",
        "deadline_s",
        "trial_deadline_s",
        "output",
    }
    if set(config) - {"parent_search"} != required or not (
        1 <= len(config["profiles"]) <= 16
        and config["parallel_trials"] in (1, 2, 3)
        and 0 < config["deadline_s"] <= 7200
        and 0 < config["trial_deadline_s"] <= 900
    ):
        raise ValueError(
            "Explicit finite acquisition search and rollout budgets required"
        )
    started = time.monotonic()
    _, seed = read(config["seed_config"])
    baseline_path, baseline = read(config["baseline_result"])
    identity_path, identity_report = read(config["identity_result"])
    expected = state_archive(baseline_path, baseline)
    zero = state_archive(identity_path, identity_report)
    if identity_report["config"].get("acquisition_profile") != identity():
        raise ValueError("Explicit identity acquisition regression required")
    if expected.keys() != zero.keys() or any(
        not np.array_equal(expected[k], zero[k]) for k in expected
    ):
        raise ValueError("Identity acquisition profile changed frozen physical states")
    if (
        seed.get("motion_condition") != "stationary_hold"
        or seed.get("finger_control_mode") != "live_contact_force"
    ):
        raise ValueError("An existing complete live-finger stationary seed is required")
    excluded = {"output", "purpose", "deadline_s"}
    if {k: v for k, v in seed.items() if k not in excluded} != {
        k: v for k, v in baseline["config"].items() if k not in excluded
    }:
        raise ValueError("Seed must match the frozen baseline controller")
    if config.get("parent_search"):
        read(config["parent_search"])
    validate_names(config["profiles"])
    for profile in config["profiles"]:
        if set(profile) != {"name", "settings"}:
            raise ValueError("Named acquisition profile settings required")
        validate(profile["settings"])
    limits = fixed_limits(baseline)
    evaluate(
        identity_report,
        baseline,
        zero["timestamps_s"],
        expected["timestamps_s"],
        limits,
    )
    out = Path(config["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    implementation = out / "implementation"
    implementation.mkdir()
    engine_path = Path(__file__).with_name("arm_replay_pilot.py")
    pinned_engine = digest(engine_path)
    replay_files = (
        "arm_replay_pilot.py",
        "acquisition_profile.py",
        "arm_finger_feedback.py",
        "arm_finger_plan.py",
        "contact_force_control.py",
        "arm_motion_condition.py",
        "arm_tracking_control.py",
        "continuous_reference.py",
        "closing_wrist_offsets.py",
    )
    pinned_replay = {
        name: digest(Path(__file__).with_name(name)) for name in replay_files
    }
    for name in set(replay_files) | {
        "acquisition_rollout_search.py",
        "continuous_retention_audit.py",
    }:
        (implementation / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(
        out / "protocol.json",
        dict(
            config=config,
            limits=limits,
            objective="sum positive normalized task violations + 0.25*(1 - full-hold bilateral load fraction) + 0.05*normalized hold rotation; original success and unsafe gates rank first",
            identity_states_exact=True,
            replay_implementations=pinned_replay,
            development_search=True,
            acceptance_changed=False,
        ),
    )
    rows = []
    baseline_score = evaluate(
        baseline, baseline, expected["timestamps_s"], expected["timestamps_s"], limits
    )

    def trial(item, suffix=""):
        name = item["name"] + suffix
        trial_config = deepcopy(seed)
        trial_config.update(
            acquisition_profile=item["settings"],
            output=str(out / name),
            purpose="Full stationary acquisition search; fixed task/reference/physics/force budgets; candidate remains development until original gates pass",
        )
        left = config["deadline_s"] - (time.monotonic() - started)
        if left <= 1:
            return dict(
                name=name,
                settings=item["settings"],
                evaluation=dict(eligible=False),
                error="search_deadline",
            )
        trial_config["deadline_s"] = min(config["trial_deadline_s"], left)
        path = out / f"{name}.json"
        write_json(path, trial_config)
        row = dict(
            name=name,
            settings=item["settings"],
            config=str(path),
            config_sha256=digest(path),
        )
        try:
            with (out / f"{name}.log").open("w") as log:
                bounded(
                    [
                        sys.executable,
                        "-m",
                        "real2sim2real.controller_actor_training.arm_replay_pilot",
                        "--config",
                        str(path),
                        "--worker",
                    ],
                    timeout=trial_config["deadline_s"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result_path = out / name / "result.json"
            result = json.loads(result_path.read_text())
            if result["implementation_sha256"] != pinned_engine:
                raise ValueError("Search mixed different replay implementations")
            for name, sha in pinned_replay.items():
                if digest(result_path.parent / "implementation" / name) != sha:
                    raise ValueError("Search mixed replay helper implementations")
            actual = state_archive(result_path, result)
            row.update(
                result=str(result_path),
                result_sha256=digest(result_path),
                success=result["success"],
                rejected_for=result["rejected_for"],
                metrics={
                    k: result[k]
                    for k in (
                        "hold_slip_m",
                        "support_wrist_drift_m",
                        "hold_minimum_clearance_m",
                        "hold_max_rotation_deg",
                        "max_robot_penetration_m",
                        "max_arm_penetration_m",
                    )
                },
                evaluation=evaluate(
                    result,
                    baseline,
                    actual["timestamps_s"],
                    expected["timestamps_s"],
                    limits,
                ),
            )
        except Exception as exc:
            row.update(
                error=type(exc).__name__ + ": " + str(exc),
                evaluation=dict(eligible=False, passed=False),
            )
            for filename in ("acquisition-path.json", "finger-path.json"):
                receipt = out / name / filename
                if receipt.exists():
                    row[filename] = json.loads(receipt.read_text())
        return row

    with ThreadPoolExecutor(max_workers=config["parallel_trials"]) as pool:
        futures = [pool.submit(trial, item) for item in config["profiles"]]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(
                out / "progress.json",
                dict(
                    completed=len(rows),
                    requested=len(futures),
                    trials=sorted(rows, key=rank),
                ),
            )
            print(
                json.dumps(
                    dict(
                        name=row["name"],
                        evaluation=row["evaluation"],
                        metrics=row.get("metrics"),
                        error=row.get("error"),
                    )
                ),
                flush=True,
            )
    ordered = sorted(rows, key=rank)
    best = next((r for r in ordered if r["evaluation"].get("eligible")), None)
    repeat = None
    promotion = False
    if best and best["evaluation"]["passed"]:
        write_json(
            out / "repeat-selection.json",
            dict(
                result=best["result"],
                sha256=best["result_sha256"],
                settings=best["settings"],
            ),
        )
        repeat = trial(dict(name=best["name"], settings=best["settings"]), "-repeat")
        if repeat["evaluation"].get("passed"):
            a = json.loads(Path(best["result"]).read_text())
            b = json.loads(Path(repeat["result"]).read_text())
            old = state_archive(Path(best["result"]), a)
            new = state_archive(Path(repeat["result"]), b)
            promotion = (
                all(np.array_equal(old[k], new[k]) for k in old)
                and a["rejected_for"] == b["rejected_for"]
            )
    report = dict(
        schema="real2sim2real.acquisition-rollout-search/1",
        config=config,
        limits=limits,
        baseline_evaluation=baseline_score,
        identity_states_exact=True,
        trials=ordered,
        best=best,
        repeat=repeat,
        stationary_candidate_reproduced=bool(promotion),
        new_moving_or_release_validated=False,
        acceptance_changed=False,
        development_search=True,
        elapsed_s=time.monotonic() - started,
        implementation_sha256=digest(implementation / "acquisition_rollout_search.py"),
        replay_implementations=pinned_replay,
    )
    write_json(out / "search.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
