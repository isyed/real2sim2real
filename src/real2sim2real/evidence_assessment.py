"""Assess an explicit set of pinned Boolean evidence claims, without promotion.

This aggregates existing reports. It does not independently establish the truth
of a report, prescribe thresholds, tune a controller or authorize hardware.
"""

import json
from pathlib import Path

from .artifacts import digest, write_json


def assess(config, base):
    checks = config["checks"]
    if not 1 <= len(checks) <= 64 or len({c["name"] for c in checks}) != len(checks):
        raise ValueError("Require 1–64 uniquely named evidence checks")
    rows = []
    for check in checks:
        path = (Path(base) / check["report"]).resolve()
        if digest(path) != check["sha256"]:
            raise ValueError("Evidence changed: " + check["name"])
        field = check["field"]
        if not isinstance(field, str) or not field:
            raise ValueError("Explicit top-level Boolean report field required")
        report = json.loads(path.read_text())
        value = report.get(field)
        if value is not None and type(value) is not bool:
            raise ValueError("Evidence claim must be Boolean or missing: " + field)
        rows.append(
            {
                **check,
                "report": str(path),
                "value": value,
                "status": "passed"
                if value is True
                else "unresolved"
                if value is None
                else "failed",
            }
        )
    return {
        "schema": "real2sim2real.evidence-assessment/1",
        "checks": rows,
        "all_required_claims_pass": all(r["value"] is True for r in rows),
        "scope": config["scope"],
        "new_physics_run": False,
        "automatic_training_or_hardware_promotion": False,
    }


def run(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    report = assess(config, path.parent)
    output = (path.parent / config["output"]).resolve()
    if output.exists():
        raise FileExistsError("Preserve assessment evidence; choose another output")
    report["protocol_sha256"] = digest(path)
    report["implementation_sha256"] = digest(Path(__file__))
    write_json(output, report)
    return output
