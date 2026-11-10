"""Replay a pinned grasp case with read-only solved-contact motion telemetry."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from real2sim2real.artifacts import digest, write_json


def run(config):
    from .object_grasp_pilot import trial
    from .contact_patch_motion import validate_settings

    validate_settings(config["diagnostic_settings"])
    path = Path(config["source_result"])
    if digest(path) != config["source_sha256"]:
        raise ValueError("Pinned contact-audit source changed")
    parent = json.loads(path.read_text())
    if digest(path.with_name("states.npz")) != parent["states_sha256"]:
        raise ValueError("Pinned contact-audit states changed")
    base = deepcopy(parent["configuration"])
    if base.get("contact_patch_motion"):
        raise ValueError("Require a source without contact-motion instrumentation")
    base["contact_patch_motion"] = config["diagnostic_settings"]
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    hashes = {}
    for name in (
        "contact_patch_audit.py",
        "contact_patch_motion.py",
        "object_grasp_pilot.py",
        "contact_force_control.py",
        "payload_pressure.py",
        "support_control.py",
    ):
        snap = out / "implementation" / name
        snap.write_bytes(Path(__file__).with_name(name).read_bytes())
        hashes[name] = digest(snap)
    result = trial(
        base,
        parent["candidate"],
        out / "replay",
        mass_scale=parent["mass_scale"],
        friction=parent["friction"],
    )
    report = dict(
        schema="real2sim2real.contact-patch-audit/1",
        config=config,
        states_identical=result["states_sha256"] == parent["states_sha256"],
        original_physical_pass=parent["success"],
        replay_physical_pass=result["success"],
        source_states_sha256=parent["states_sha256"],
        replay_result_sha256=digest(out / "replay/result.json"),
        summary=result["contact_patch_motion_summary"],
        diagnostics_cannot_promote_source=True,
        implementation_sha256=hashes,
    )
    write_json(out / "audit.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
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
            timeout=600,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only contact diagnostic required")
    run(config)


if __name__ == "__main__":
    main()
