"""Compare saved arm-replay evidence without rerunning physics or changing gates."""

import argparse
import html
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from real2sim2real.artifacts import digest, write_json


def summarize(report):
    hold = [r for r in report["records"] if r["phase"] == "hold"]
    reference = [r for r in hold if "reference_position_error_m" in r]
    result = {
        k: report[k]
        for k in (
            "success",
            "rejected_for",
            "hold_slip_m",
            "support_wrist_drift_m",
            "hold_minimum_clearance_m",
            "hold_opposed_contact_fraction",
            "support_load_fraction",
            "max_robot_penetration_m",
            "max_arm_penetration_m",
            "maximum_dynamic_velocity_ratio",
            "minimum_dynamic_joint_margin_rad",
        )
    }
    result.update(
        maximum_reference_position_error_m=max(
            (r["reference_position_error_m"] for r in reference), default=None
        ),
        maximum_reference_rotation_error_deg=max(
            (r["reference_rotation_error_deg"] for r in reference), default=None
        ),
        maximum_support_load_offset_m=max(
            (
                r.get("object_feedback", {}).get("support_load_offset_m", 0)
                for r in hold
            ),
            default=0,
        ),
        object_feedback=report.get("object_feedback"),
        wrist_tracking_by_phase=report.get("wrist_tracking_by_phase"),
        commanded_wrist_tracking_by_phase=report.get(
            "commanded_wrist_tracking_by_phase"
        ),
        saturation_fraction=report["saturation_fraction"],
    )
    return result


def compare(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists() or (output / "review.html").exists():
        raise ValueError("Use a new comparison directory")
    paths = [Path(p).resolve() for p in paths]
    if len(paths) < 2 or len(paths) > 24 or len(set(paths)) != len(paths):
        raise ValueError("Compare 2–24 distinct saved results")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if report["schema"] != "real2sim2real.arm-replay-pilot/1":
            raise ValueError("Expected an arm-replay report")
        for file, key in (
            ("states.npz", "states_sha256"),
            ("scene.xml", "scene_sha256"),
        ):
            if digest(path.with_name(file)) != report[key]:
                raise ValueError("Saved rollout artifact changed")
        option = ET.parse(path.with_name("scene.xml")).getroot().find("option")
        actual_step = (
            float(option.get("timestep", "0.002")) if option is not None else 0.002
        )
        rows.append(
            dict(
                name=path.parent.name,
                result=str(path),
                result_sha256=digest(path),
                config=report["config"],
                integration_timestep_s=actual_step,
                metrics=summarize(report),
            )
        )
    if len({r["config"]["audit_sha256"] for r in rows}) != 1:
        raise ValueError("Comparison must preserve the audited task and path")
    defaults = dict(
        timestep_s=0.002,
        velocity_feedforward=False,
        exclude_adjacent_arm_collisions=False,
    )
    keys = (
        "placement",
        "kp_nm_per_rad",
        "kd_nm_s_per_rad",
        "torque_limits_nm",
        *defaults,
    )
    differing = [
        k
        for k in keys
        if len(
            {
                json.dumps(
                    r["integration_timestep_s"]
                    if k == "timestep_s"
                    else r["config"].get(k, defaults.get(k)),
                    sort_keys=True,
                )
                for r in rows
            }
        )
        > 1
    ]
    result = dict(
        schema="real2sim2real.arm-replay-comparison/1",
        rows=rows,
        differing_base_control_settings=differing,
        physics_rerun=False,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(output / "comparison.json", result)
    headers = [
        "Run",
        "Primary slip (mm)",
        "Support drift (mm)",
        "Clearance (mm)",
        "Opposed contact (%)",
        "Support load (%)",
        "Reference error (mm)",
        "Rejections",
    ]
    body = []
    for row in rows:
        m = row["metrics"]
        path = Path(row["result"])
        link = (
            path.with_name("review.html")
            if path.with_name("review.html").exists()
            else path
        )
        relative = os.path.relpath(link, output)
        values = [
            f'<a href="{html.escape(relative, quote=True)}">{html.escape(row["name"])}</a>'
        ]
        for key, factor in (
            ("hold_slip_m", 1000),
            ("support_wrist_drift_m", 1000),
            ("hold_minimum_clearance_m", 1000),
            ("hold_opposed_contact_fraction", 100),
            ("support_load_fraction", 100),
            ("maximum_reference_position_error_m", 1000),
        ):
            values.append("unavailable" if m[key] is None else f"{m[key] * factor:.2f}")
        values.append(html.escape(", ".join(m["rejected_for"]) or "none"))
        body.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Wrist tracking and load sharing</title>'
        "<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}pre{white-space:pre-wrap}</style>"
        "<h1>Wrist tracking and load sharing</h1><p>Saved simulated dynamics. No hardware or policy-readiness claim. "
        "The contact-isolated condition cannot count as task success. Reference fidelity remains an independent acceptance gate.</p>"
        "<p>Different arm gains, limits, placement or integration settings: "
        + html.escape(", ".join(differing) or "none")
        + ".</p>"
        "<table><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr>"
        + "".join(body)
        + "</table>"
        '<p>Open a run for video where available. <a href="comparison.json">Metrics, configuration and provenance</a>.</p>'
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.results, args.output)


if __name__ == "__main__":
    main()
