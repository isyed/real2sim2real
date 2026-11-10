"""Pinned, bounded paired-contact force diagnostic; never promotes a robot plan."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from real2sim2real.retargeting.paired_contacts import paired_wrench_screen


def diagnose(config):
    plan_path, audit_path = Path(config["plan"]), Path(config["audit"])
    for key, path in [("plan", plan_path), ("audit", audit_path)]:
        if digest(path) != config[key + "_sha256"]:
            raise ValueError("Pinned diagnostic input changed")
    plan, audit = json.loads(plan_path.read_text()), json.loads(audit_path.read_text())
    if audit["config"]["plan_sha256"] != config["plan_sha256"]:
        raise ValueError("Observation audit belongs to a different plan")
    ids = config["candidate_ids"]
    if not 2 <= len(ids) <= 16 or len(set(ids)) != len(ids):
        raise ValueError("Two to sixteen distinct candidate IDs required")
    candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    evidence = {c["candidate_id"]: c for c in audit["candidates"]}
    if not set(ids) <= candidates.keys() & evidence.keys():
        raise ValueError("Every candidate requires plan and observation evidence")
    rows = []
    for a, b in itertools.combinations(ids, 2):
        ca, cb = candidates[a], candidates[b]
        com = np.asarray(ca["pair"]["center_of_mass_m"])
        if not np.allclose(com, cb["pair"]["center_of_mass_m"], atol=1e-9, rtol=0):
            raise ValueError("Contact sets must share one center of mass and world")
        # Opposite sides of the COM in the horizontal plane, not just opposing
        # normals on the two faces of a single thin wall.
        va = np.asarray(ca["pair"]["center_m"])[:2] - com[:2]
        vb = np.asarray(cb["pair"]["center_m"])[:2] - com[:2]
        if va @ vb >= 0:
            continue
        assumptions = ca["pad_plan"]["point_wrench_baseline"]
        keys = ("mass_kg", "friction", "max_normal_force_n")
        if any(
            assumptions[k] != cb["pad_plan"]["point_wrench_baseline"][k] for k in keys
        ):
            raise ValueError("Paired screen cannot change physical assumptions")
        wrench = paired_wrench_screen(
            ca["pad_plan"]["sampled_patches"],
            cb["pad_plan"]["sampled_patches"],
            com,
            **{k: assumptions[k] for k in keys},
        )
        supported = all(evidence[i]["contact_region_supported"] for i in (a, b))
        rows.append(
            dict(
                candidate_ids=[a, b],
                gravity_wrench=wrench,
                observation_regions_supported=supported,
                ready_for_robot_pair_fitting=bool(wrench["feasible"] and supported),
                control_ready=False,
            )
        )
    report = dict(
        schema="real2sim2real.paired-contact-diagnostic/1",
        config=config,
        pairs=rows,
        source_plan_changed=False,
        robot_pair_validated=False,
        control_ready=False,
        scope="Contact arrangement only; input hand poses are separate single-hand hypotheses.",
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "diagnostic.json", report)
    return report


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
        diagnose(config)
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
