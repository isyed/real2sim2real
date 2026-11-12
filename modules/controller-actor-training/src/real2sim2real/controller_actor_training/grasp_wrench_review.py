"""Compare pinned instantaneous grasp audits and saved wrist motion, read-only."""

import argparse
import html
import json
from pathlib import Path

from real2sim2real.artifacts import digest, write_json

from .paired_acquisition_audit import pinned_json
from .sampled_motion_variation import position_variation


def summarize(audit):
    result = {}
    for stage in ("hands_only", "external_support"):
        rows = [r for r in audit["rows"] if r["contact_stage"] == stage]
        if not rows:
            continue
        envelopes = {}
        for key in rows[0]["envelopes"]:
            envelopes[key] = {}
            for cone in ("inner", "outer"):
                values = [r["envelopes"][key][cone] for r in rows]
                factors = [
                    v["capacity"]["capacity_factor"]
                    for v in values
                    if v["capacity"].get("verified")
                ]
                envelopes[key][cone] = {
                    "verified_minimum": sum(
                        bool(v["minimum"].get("verified")) for v in values
                    ),
                    "infeasible_times_s": [
                        r["time_s"]
                        for r, v in zip(rows, values)
                        if v["minimum"]["status"] == 2
                    ],
                    "unresolved_times_s": [
                        r["time_s"]
                        for r, v in zip(rows, values)
                        if v["minimum"]["status"] not in (0, 2)
                        or (
                            v["minimum"]["status"] == 0
                            and not v["minimum"].get("verified")
                        )
                    ],
                    "exact_cone_and_feedback_ball_witness": sum(
                        bool(
                            v["minimum"].get("verified")
                            and v["minimum"].get("exact_cone_satisfied")
                            and v["minimum"].get("actual_feedback_ball_satisfied")
                        )
                        for v in values
                    ),
                    "minimum_solved_maximum_load_factor": min(factors)
                    if factors
                    else None,
                    "capacity_unsolved": len(values) - len(factors),
                }
        result[stage] = {
            "samples": len(rows),
            "envelopes": envelopes,
            "hand_vertical_force_range_n": [
                min(r["validation"]["measured_hand_wrench"][2] for r in rows),
                max(r["validation"]["measured_hand_wrench"][2] for r in rows),
            ],
        }
    return result


def run(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for item in config["cases"]:
        audit = pinned_json(item["audit"])
        for source, sha in audit["sources"].items():
            if digest(Path(source)) != sha:
                raise ValueError(
                    "Pinned audit implementation or physical source changed"
                )
        source = pinned_json(audit["config"]["result"])
        rows = source["records"]
        motion = {}
        for window in config["motion_windows"]:
            selected = [
                r
                for r in rows
                if window["start_s"] - 1e-8 <= r["time_s"] <= window["end_s"] + 1e-8
            ]
            motion[window["label"]] = {
                role: position_variation(
                    [r["time_s"] for r in selected],
                    [r[field] for r in selected],
                    trend_window_s=config["trend_window_s"],
                )
                for role, field in [
                    ("primary", "wrist_position_m"),
                    ("support", "support_wrist_position_m"),
                ]
            }
        first_touch = next(
            (
                r["time_s"]
                for r in rows
                if r["time_s"] >= audit["config"]["start_s"] and r["table_contacts"]
            ),
            None,
        )
        witness = min(
            audit["rows"],
            key=lambda r: abs(r["time_s"] - audit["config"]["witness_time_s"]),
        )
        cases.append(
            {
                "label": item["label"],
                "audit": item["audit"],
                "capacity": summarize(audit),
                "first_sampled_table_contact_s": first_touch,
                "force_witness": dict(
                    time_s=witness["time_s"], **witness["validation"]
                ),
                "motion_variation": motion,
                "source_success": source["success"],
                "source_rejected_for": source["rejected_for"],
            }
        )
    summary = {
        "schema": "grasp-wrench-review/1",
        "config": config,
        "cases": cases,
        "physics_rerun": False,
        "training_run": False,
        "controller_promoted": False,
        "implementation_sha256": digest(Path(__file__)),
        "motion_implementation_sha256": digest(
            Path(__file__).with_name("sampled_motion_variation.py")
        ),
    }
    write_json(out / "summary.json", summary)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Grasp load audit</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:15px}td,th{padding:9px;border:1px solid #ccc;text-align:left}table{border-collapse:collapse}code{white-space:pre-wrap}</style><h1>Grasp force-and-torque audit</h1>',
        "<p>Read-only saved-state analysis. No controller or acceptance limit changed. Force balance is an instantaneous ideal-contact test, not a trajectory or hardware certificate. Inner cones provide sufficient ideal friction solutions; outer cones only bound possibility. Maximum load factors need not imply feasibility at the nominal load because torque envelopes also have lower bounds.</p>",
        "<p>Hands-only and table-supported samples are separated. At table-supported samples, the LP still asks counterfactually whether hands alone could support the requested load. Measured table forces are never granted to that LP. Normal caps of 10/25 N per anatomical group are analytic envelopes, not manufacturer specifications.</p>",
    ]
    for case in cases:
        parts.append("<h2>" + html.escape(case["label"]) + "</h2>")
        parts.append(
            f"<p>First recorded table contact after descent starts: {case['first_sampled_table_contact_s']:.3f} s. Source success remains {case['source_success']}.</p>"
        )
        w = case["force_witness"]
        parts.append(
            f"<p>At {w['time_s']:.3f} s: hands vertical {w['measured_hand_wrench'][2]:+.2f} N; external support vertical {w['external_object_wrench'][2]:+.2f} N; actual required net vertical {w['actual_required_wrench'][2]:+.2f} N. Positive is upward.</p>"
        )
        parts.append(
            "<table><tr><th>Stage</th><th>Envelope/demand</th><th>Inner-cone feasible samples</th><th>Outer-cone infeasible samples</th></tr>"
        )
        for stage, content in case["capacity"].items():
            for key, cones in content["envelopes"].items():
                parts.append(
                    f"<tr><td>{stage}</td><td>{key}</td><td>{cones['inner']['verified_minimum']}/{content['samples']}</td><td>{len(cones['outer']['infeasible_times_s'])}/{content['samples']}</td></tr>"
                )
        parts.append(
            "</table><h3>Saved wrist position variation</h3><p>RMS residual after removing a local cubic position trend over approximately 0.3 s, excluding filter edges. This includes deliberate motion and is not tracking error or proof of oscillation. Saved 50 Hz samples cannot resolve motion above 25 Hz. Rendering does not create these positional variations.</p><ul>"
        )
        for window, roles in case["motion_variation"].items():
            parts.append(
                "<li>"
                + html.escape(window)
                + ": "
                + "; ".join(
                    f"{role} RMS {v['rms_position_variation_mm']:.3f} mm, peak {v['peak_position_variation_mm']:.3f} mm"
                    for role, v in roles.items()
                )
                + "</li>"
            )
        parts.append("</ul>")
    parts.append(
        '<p>Read <a href="summary.json">all summary data</a>. Inscribed/outer feedback boxes bound the existing per-pad 0.2 rad joint-offset ball; failure of the inner box alone does not prove the actual controller budget is insufficient. Motor solutions do not establish contact compliance, sliding compatibility, stable pressure control, acquisition or release. Archived forces are coherently re-solved with zero warmstart; they are not additional physical rollouts.</p>'
    )
    (out / "review.html").write_text("\n".join(parts))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
