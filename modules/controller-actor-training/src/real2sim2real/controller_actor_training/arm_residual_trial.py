"""Bounded CPU pressure-policy trial under a pinned arm controller's task gates."""

import argparse
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json

from .arm_replay_comparison import summarize
from .arm_residual_env import ArmResidualEnv
from .pressure_policy import PressurePolicy


def matched_protocol(baseline, candidate):
    """Check physical protocol independently of the residual's reward or policy."""
    configs = [dict(r["config"]) for r in (baseline, candidate)]
    for config in configs:
        config.pop("output")
    return dict(
        frozen_config_equal=configs[0] == configs[1],
        physical_scene_equal=baseline["scene_sha256"] == candidate["scene_sha256"],
        controller_clock_equal=all(
            baseline[k] == candidate[k]
            for k in ("controller_period_s", "physics_timestep_s")
        ),
        complete_timeline=(
            len(baseline["records"]) == len(candidate["records"])
            and all(
                a["time_s"] == b["time_s"] and a["phase"] == b["phase"]
                for a, b in zip(baseline["records"], candidate["records"])
            )
            and baseline["controller_updates"] == candidate["controller_updates"]
            and "incomplete_pilot" not in candidate["rejected_for"]
        ),
    )


def run(config):
    if set(config) != {"environment", "policy"}:
        raise ValueError("Explicit residual environment and pressure policy required")
    policy = PressurePolicy(config["policy"])
    env = ArmResidualEnv(config["environment"])
    output = Path(config["environment"]["output"])
    # Pin the actual policy implementation before any simulation starts.
    implementation = Path(__file__).with_name("pressure_policy.py")
    (output / "pressure_policy.py").write_bytes(implementation.read_bytes())
    write_json(output / "trial-config.json", config)
    try:
        observation, info = env.reset()
        steps = 0
        while True:
            observation, _, terminated, truncated, info = env.step(
                policy(observation, info)
            )
            steps += 1
            if steps % 250 == 0:
                print(
                    json.dumps(
                        dict(
                            policy_steps=steps,
                            time_s=info["time_s"],
                            phase=info["phase"],
                        )
                    ),
                    flush=True,
                )
            if truncated:
                raise RuntimeError("Infrastructure-truncated trial is ineligible")
            if terminated:
                break
        report = env.report
        checks = matched_protocol(env.baseline, report)
        summary = dict(
            schema="real2sim2real.arm-pressure-policy-trial/1",
            config=config,
            trial_config_sha256=digest(output / "trial-config.json"),
            policy_implementation_sha256=digest(output / "pressure_policy.py"),
            baseline_sha256=config["environment"]["baseline_sha256"],
            result=str(env.directory / "result.json"),
            result_sha256=digest(env.directory / "result.json"),
            protocol_checks=checks,
            complete_matched_trial=all(checks.values()),
            acceptance_passed=all(checks.values()) and report["success"],
            metrics=summarize(report),
            peak_pressure_offset_n=report["residual_adapter"][
                "peak_requested_pressure_offset_n"
            ],
            policy_actions=report["residual_adapter"]["policy_actions"],
            training_run=False,
            hardware_ready=False,
        )
        write_json(output / "trial.json", summary)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in ("acceptance_passed", "complete_matched_trial", "metrics")
                }
            ),
            flush=True,
        )
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        return run(config)
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
        timeout=min(config["environment"]["deadline_s"], 900),
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )


if __name__ == "__main__":
    main()
