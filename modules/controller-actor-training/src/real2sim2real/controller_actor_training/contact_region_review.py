"""Static browser review of contact evidence and paired force diagnostics."""

import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_review(audit_paths, paired_path, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audits = [(Path(p), json.loads(Path(p).read_text())) for p in audit_paths]
    if not audits:
        raise ValueError("At least one contact audit required")
    paired = json.loads(Path(paired_path).read_text())
    plan_hash = paired["config"]["plan_sha256"]
    if any(a["config"]["plan_sha256"] != plan_hash for _, a in audits):
        raise ValueError("Review inputs must use the same contact plan")
    chunks = [
        '<!doctype html><meta charset="utf-8"><title>Contact evidence review</title>',
        "<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;padding:20px;background:#17202b;color:#eee}table{border-collapse:collapse;width:100%}td,th{border:1px solid #637080;padding:8px}img{max-width:100%}a{color:#6ddcff}</style>",
        "<h1>Contact evidence and paired planning</h1>",
        "<p>Static observation and force diagnostics. No two-hand robot grasp or lift is validated. Depth is reused reconstruction input, not independent ground truth.</p>",
    ]
    fmt = lambda x: "missing" if x is None else f"{1000 * x:.2f}"
    for path, audit in audits:
        width = 2 * audit["config"]["image_sampling"]["radius_pixels"] + 1
        chunks.append(
            f"<h2>{width} × {width} depth window</h2><table><tr><th>Candidate</th><th>Proxy boundary error (mm)</th><th>Contact</th><th>Observed frames</th><th>Median / p90 depth error (mm)</th><th>Evidence</th></tr>"
        )
        for candidate in audit["candidates"]:
            for contact in candidate["contacts"]:
                s = contact["summary"]
                values = [
                    candidate["candidate_id"],
                    fmt(candidate["local_proxy"]["boundary_error_m"]),
                    contact["contact_index"],
                    s["observed_frames"],
                    f"{fmt(s['median_abs_depth_error_m'])} / {fmt(s['p90_abs_depth_error_m'])}",
                    s["evidence"],
                ]
                chunks.append(
                    "<tr>"
                    + "".join(f"<td>{escape(str(v))}</td>" for v in values)
                    + "</tr>"
                )
        chunks.append("</table>")
    chunks.append(
        "<h2>Opposite-side contact sets</h2><p>These combine separate single-hand hypotheses. Simultaneous robot reachability, collisions and actuator feasibility remain unchecked.</p><table><tr><th>Pair</th><th>Static gravity balance</th><th>Observed regions supported</th><th>Ready for robot pair fitting</th></tr>"
    )
    for row in paired["pairs"]:
        values = [
            row["candidate_ids"],
            row["gravity_wrench"]["feasible"],
            row["observation_regions_supported"],
            row["ready_for_robot_pair_fitting"],
        ]
        chunks.append(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
        )
    chunks.append(
        "</table><h2>Projected contact observations</h2><p>Cyan: depth sampled after visibility checks; orange: unobserved. Cyan does not mean the depth agrees. Labels are candidate:contact index.</p>"
    )
    first_path, first = audits[0]
    for frame in first["config"]["frames"]:
        relative = quote(
            os.path.relpath(first_path.parent / f"frame-{frame:06d}.jpg", output.parent)
        )
        chunks.append(
            f'<figure><img loading="lazy" src="{relative}" alt="Contact projections, frame {frame}"><figcaption>Frame {frame}</figcaption></figure>'
        )
    output.write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_review(args.audit, args.paired, args.output)


if __name__ == "__main__":
    main()
