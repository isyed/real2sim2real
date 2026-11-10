"""Combine observation-supported sites without assuming they are robot grasps."""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from real2sim2real.retargeting.pad_contacts import gravity_wrench_screen
from real2sim2real.retargeting.contact_site_search import upward_force_capacity


def combine(config):
    if not 1 <= len(config["surveys"]) <= 8:
        raise ValueError("One to eight pinned surveys required")
    reports = []
    for record in config["surveys"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Surface survey changed")
        reports.append(json.loads(path.read_text()))
    source = reports[0]["config"]
    for report in reports:
        if any(
            report["config"][k] != source[k]
            for k in [
                "plan_sha256",
                "capture_sha256",
                "calibration_sha256",
                "observation_limits",
            ]
        ):
            raise ValueError("Cannot combine different scenes or observation criteria")
    if digest(Path(source["plan"])) != source["plan_sha256"]:
        raise ValueError("Plan changed")
    plan = json.loads(Path(source["plan"]).read_text())
    candidate = plan["candidates"][0]
    physics = candidate["pad_plan"]["point_wrench_baseline"]
    sites = []
    seen = set()
    for report_index, report in enumerate(reports):
        for site in report["sites"]:
            if (
                not site["shortlisted"]
                or site["discovery"]["evidence"] != "supported"
                or site["validation"]["evidence"] != "supported"
                or site["face_index"] in seen
            ):
                continue
            seen.add(site["face_index"])
            sites.append(dict(survey_index=report_index, **site))
    force = dict(feasible=False, reason="fewer_than_two_supported_sites")
    cap = 0.0
    if sites:
        normals = np.array([s["normal_task_world"] for s in sites])
        points = np.array([s["point_task_world_m"] for s in sites])
        cap = float(
            np.maximum(0, upward_force_capacity(normals, physics["friction"])).sum()
            * physics["max_normal_force_n"]
        )
        if len(sites) >= 2:
            force = gravity_wrench_screen(
                points,
                normals,
                candidate["pair"]["center_of_mass_m"],
                **{
                    k: physics[k] for k in ["mass_kg", "friction", "max_normal_force_n"]
                },
            )
    result = dict(
        schema="real2sim2real.observed-support-arrangement/1",
        config=config,
        sites=sites,
        gravity_wrench=force,
        optimistic_upward_force_bound_n=cap,
        object_weight_n=physics["mass_kg"] * 9.81,
        bound_ignores_horizontal_balance_and_torque=True,
        per_site_normal_force_limit_n=physics["max_normal_force_n"],
        reconstructed_normals_not_independently_validated=True,
        finite_patch_validated=False,
        robot_pair_validated=False,
        control_ready=False,
    )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "arrangement.json", result)
    return result


def main():
    import os
    import sys
    from real2sim2real.bounded_process import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        combine(config)
    else:
        return run(
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


if __name__ == "__main__":
    main()
