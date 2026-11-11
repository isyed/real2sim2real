"""Bounded CPU zero-residual verification against pinned frozen physical evidence."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .arm_residual_env import ArmResidualEnv


def compare(baseline_path, candidate_path):
    baseline_path, candidate_path = Path(baseline_path), Path(candidate_path)
    baseline, candidate = [
        json.loads(p.read_text()) for p in (baseline_path, candidate_path)
    ]
    for path, report in ((baseline_path, baseline), (candidate_path, candidate)):
        if digest(path.with_name("states.npz")) != report["states_sha256"]:
            raise ValueError("Physical state archive changed")
        if digest(path.with_name("scene.xml")) != report["scene_sha256"]:
            raise ValueError("Physical scene changed")
    config_a, config_b = [dict(r["config"]) for r in (baseline, candidate)]
    config_a.pop("output")
    config_b.pop("output")
    adapter = candidate.get("residual_adapter", {})
    if config_a != config_b or adapter.get("all_actions_zero") is not True:
        raise ValueError(
            "Exact frozen configuration and zero requested actions required"
        )
    trace_path = candidate_path.with_name("policy-trace.npz")
    if digest(trace_path) != adapter["trace_sha256"]:
        raise ValueError("Policy trace changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        if (
            not len(trace["actions"])
            or trace["actions"].shape[1:] != (4,)
            or np.any(trace["actions"])
        ):
            raise ValueError("Complete nonempty zero-action trace required")
        if len(trace["observations"]) != len(trace["actions"]) + 1:
            raise ValueError("Every policy transition needs a terminal observation")
    arrays = {}
    with (
        np.load(baseline_path.with_name("states.npz"), allow_pickle=False) as a,
        np.load(candidate_path.with_name("states.npz"), allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files):
            raise ValueError("Physical archive schemas differ")
        for name in a.files:
            same_shape = a[name].shape == b[name].shape
            arrays[name] = dict(
                equal=bool(np.array_equal(a[name], b[name])),
                baseline_shape=list(a[name].shape),
                candidate_shape=list(b[name].shape),
                max_absolute_difference=float(np.max(abs(a[name] - b[name]), initial=0))
                if same_shape
                else None,
            )
    fields = {k for k in baseline if k.startswith(("hold_", "support_"))}
    fields.update(
        (
            "success",
            "rejected_for",
            "controller_updates",
            "controller_period_s",
            "physics_timestep_s",
            "records",
            "max_robot_penetration_m",
            "max_arm_penetration_m",
            "maximum_dynamic_velocity_ratio",
            "minimum_dynamic_joint_margin_rad",
            "solver_warning_counts",
            "torque_peak_nm",
            "live_finger_feedback_final",
        )
    )
    equality = {k: baseline[k] == candidate[k] for k in sorted(fields)}
    for k in (
        "continuous_protocol",
        "continuous_metrics",
        "motion_condition",
        "demonstrated_motion_executed",
        "stationary_admission",
    ):
        equality[k] = baseline.get(k) == candidate.get(k)
    return dict(
        schema="real2sim2real.arm-zero-residual-verification/1",
        baseline_result=str(baseline_path),
        baseline_sha256=digest(baseline_path),
        candidate_result=str(candidate_path),
        candidate_sha256=digest(candidate_path),
        state_arrays=arrays,
        report_field_equality=equality,
        scene_equal=baseline["scene_sha256"] == candidate["scene_sha256"],
        zero_reproduction_passed=all(x["equal"] for x in arrays.values())
        and all(equality.values())
        and baseline["scene_sha256"] == candidate["scene_sha256"],
        baseline_success=baseline["success"],
        candidate_success=candidate["success"],
        rejected_for=candidate["rejected_for"],
        task_acceptance_unchanged=True,
        training_run=False,
        hardware_ready=False,
    )


def run(config):
    env = ArmResidualEnv(config)
    try:
        env.reset()
        while True:
            _, _, terminated, truncated, _ = env.step(np.zeros(env.action_dim))
            if truncated:
                raise RuntimeError(
                    "Truncated zero replay cannot establish reproduction"
                )
            if terminated:
                break
        verification = compare(config["baseline_result"], env.directory / "result.json")
        write_json(Path(config["output"]) / "verification.json", verification)
        print(
            json.dumps(
                {
                    k: verification[k]
                    for k in (
                        "zero_reproduction_passed",
                        "baseline_success",
                        "candidate_success",
                        "rejected_for",
                    )
                }
            ),
            flush=True,
        )
        if not verification["zero_reproduction_passed"]:
            raise RuntimeError("Zero residual changed the frozen rollout; do not train")
        return verification
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=min(config["deadline_s"], 900),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    main()
