"""Read-only retention timeline using coherent pad telemetry and measured wrists."""

import argparse
import html
import json
from pathlib import Path

from real2sim2real.artifacts import digest, write_json
from .acquisition_compare import describe, relative_object, pose_difference
from .contact_patch_motion import summarize


def retention_timeline(records, settings):
    acquisition = describe(records, settings)
    hold = [r for r in records if r["phase"] == "hold"]
    result = []
    # These are diagnostic samples. Original source acceptance is retained below.
    for row in hold:
        item = dict(
            time_s=row["time_s"],
            relative_change_from_first_hold={
                role: pose_difference(
                    relative_object(row, role), relative_object(hold[0], role)
                )
                for role in ("primary", "secondary")
            },
            reference_position_error_m=row.get("reference_position_error_m"),
            reference_rotation_error_deg=row.get("reference_rotation_error_deg"),
            wrist_tracking=row.get("wrist_tracking"),
            frozen_pad_counts={
                k: v["frozen_steps"] for k, v in row["contact_force_feedback"].items()
            },
        )
        result.append(item)
    midpoint = hold[0]["time_s"] + (hold[-1]["time_s"] - hold[0]["time_s"]) / 2
    windows = {
        "early_hold": [r for r in hold if r["time_s"] <= midpoint],
        "late_hold": [r for r in hold if r["time_s"] > midpoint],
    }
    return dict(
        acquisition=acquisition,
        timeline=result,
        windows={name: summarize(rows, settings) for name, rows in windows.items()},
        threshold_anchor_note="Diagnostic first-hold changes do not replace original acceptance anchors or verdicts.",
    )


def audit(paths, output):
    if not 1 <= len(paths) <= 24:
        raise ValueError("One to 24 saved arm runs required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for item in paths:
        path = Path(item).resolve()
        source = json.loads(path.read_text())
        if source["schema"] != "real2sim2real.arm-replay-pilot/1":
            raise ValueError("Arm-replay source required")
        for name, key in (
            ("states.npz", "states_sha256"),
            ("scene.xml", "scene_sha256"),
        ):
            if digest(path.with_name(name)) != source[key]:
                raise ValueError("Saved arm artifacts changed")
        result = retention_timeline(
            source["records"], source["config"]["contact_patch_motion"]
        )
        reports.append(
            dict(
                name=path.parent.name,
                result=str(path),
                result_sha256=digest(path),
                source_success=source["success"],
                source_rejected_for=source["rejected_for"],
                controller_period_s=source["controller_period_s"],
                physics_timestep_s=source["physics_timestep_s"],
                **result,
            )
        )
    report = dict(
        schema="real2sim2real.arm-retention-audit/1",
        runs=reports,
        diagnostic_only=True,
        source_verdicts_unchanged=True,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "audit.json", report)
    cards = []
    for run in reports:
        phase = run["acquisition"]["phases"]["hold"]
        rows = "".join(
            "<tr><td>"
            + html.escape(k)
            + "</td><td>"
            + f"{v['mean_coherent_force_n']:.2f}"
            + "</td><td>"
            + f"{v['sampled_sliding_path_m'] * 1000:.2f}"
            + "</td><td>"
            + f"{v['maximum_offset_norm_rad']:.3f}"
            + "</td></tr>"
            for k, v in phase["pads"].items()
        )
        # Inline SVG preserves portability without introducing a plotting dependency.
        timeline = run["timeline"]
        t0 = timeline[0]["time_s"]
        duration = timeline[-1]["time_s"] - t0
        lines = []
        for role, color in (("primary", "#b22"), ("secondary", "#26a")):
            points = " ".join(
                f"{40 + (r['time_s'] - t0) / max(duration, 1e-9) * 700:.2f},{230 - min(r['relative_change_from_first_hold'][role]['translation_m'] * 1000, 20) / 20 * 200:.2f}"
                for r in timeline
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        svg = (
            '<svg viewBox="0 0 800 270" width="800" role="img" aria-label="Relative grasp drift over hold"><line x1="40" x2="740" y1="130" y2="130" stroke="#888" stroke-dasharray="5"/>'
            + "".join(lines)
            + '<text x="40" y="255">Hold time →; vertical: 0–20 mm. Dashed: 10 mm. Red: primary, blue: support.</text></svg>'
        )
        cards.append(
            "<h2>"
            + html.escape(run["name"])
            + "</h2><p>Source verdict: "
            + html.escape(str(run["source_rejected_for"] or "pass"))
            + "</p>"
            + svg
            + "<table><tr><th>Pad</th><th>Mean force N</th><th>Sampled sliding path mm</th><th>Peak joint-offset norm rad</th></tr>"
            + rows
            + "</table>"
        )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Arm retention audit</title><style>body{font-family:system-ui;max-width:1100px;margin:auto}td,th{padding:6px;text-align:left}svg{max-width:100%}</style><h1>Arm retention diagnostics</h1><p>Coherent contact forces are re-solved on separate data at saved states. Sliding paths are sampled estimates; patch-centroid migration is a different measurement. Curves use the first hold sample; original source acceptance verdicts remain unchanged.</p>'
        + "".join(cards)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.results, args.output)


if __name__ == "__main__":
    main()
