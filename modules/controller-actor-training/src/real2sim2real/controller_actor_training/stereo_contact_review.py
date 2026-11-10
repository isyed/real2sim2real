"""Browser review of direct stereo contact evidence and matcher controls."""

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
from statistics import median
from urllib.parse import quote


def write_review(paths, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = [(Path(p), json.loads(Path(p).read_text())) for p in paths]
    if not reports:
        raise ValueError("At least one stereo contact audit required")
    first = reports[0][1]
    for _, report in reports:
        for key in [
            "contact_audit_sha256",
            "calibration_sha256",
            "capture_sha256",
            "fit_frames",
            "validation_frames",
        ]:
            if report["config"][key] != first["config"][key]:
                raise ValueError(
                    "Sensitivity reviews must share sources, calibration and frame splits"
                )
    chunks = [
        '<!doctype html><meta charset="utf-8"><title>Direct stereo contact evidence</title>',
        "<style>body{font:17px sans-serif;max-width:1500px;margin:25px auto;padding:15px;background:#17202b;color:#eee}td,th{padding:8px;border:1px solid #738193}table{border-collapse:collapse}img{width:100%}a{color:#6ddcff}</style>",
        "<h1>Direct stereo contact evidence</h1>",
        "<p>Fixed geometry and calibration. Image locations are model-seeded; matches are consistency checks, not independent surface truth. No geometry correction, robot grasp or control promotion was performed.</p>",
    ]
    for path, report in reports:
        width = 2 * report["config"]["matching"]["radius"] + 1
        rows = [r for c in report["contacts"] for r in c["observations"]]
        reasons = Counter(r["reason"] for r in rows)
        controls = report["textured_image_controls"]
        good = sum(c["match"]["accepted"] for c in controls)
        wall = [r["projected_wall_width_px"] for r in rows]
        sensitivity = [
            r["depth_change_per_disparity_pixel_m"] * 1000
            for r in rows
            if r.get("depth_change_per_disparity_pixel_m") is not None
        ]
        chunks += [
            f"<h2>{width} × {width} matching patch</h2>",
            f"<p>Textured-image controls accepted: {good}/{len(controls)}. Contact matches accepted: {sum(r['observed'] for r in rows)}/{len(rows)}.</p>",
            f"<p>Projected wall width: {min(wall):.2f}–{max(wall):.2f} px; median {median(wall):.2f} px.</p>",
        ]
        if sensitivity:
            chunks.append(
                f"<p>Predicted depth change per disparity pixel: {min(sensitivity):.2f}–{max(sensitivity):.2f} mm. This is sensitivity, not calibrated uncertainty.</p>"
            )
        chunks.append(
            "<ul>"
            + "".join(
                f"<li>{escape(reason)}: {count}</li>"
                for reason, count in sorted(reasons.items())
            )
            + "</ul>"
        )
        chunks.append(
            "<table><tr><th>Candidate/contact</th><th>Fit matches</th><th>Held-out matches</th><th>Held-out evidence</th></tr>"
        )
        for c in report["contacts"]:
            values = [
                f"{c['candidate_id']}:{c['contact_index']}",
                c["summaries"]["fit"]["observed_frames"],
                c["summaries"]["validation"]["observed_frames"],
                c["summaries"]["validation"]["evidence"],
            ]
            chunks.append(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
            )
        chunks.append("</table>")
        link = quote(os.path.relpath(path, output.parent))
        chunks.append(f'<p><a href="{link}">Full audit JSON</a></p>')
    chunks.append(
        "<h2>Three-view crops</h2><p>RGB target, rectified left, rectified right. Orange: rejected/missing evidence; cyan: accepted correspondence. Green marks an observed right-image match when present. Crops are enlarged for inspection; enlargement creates no new detail.</p>"
    )
    for frame in sorted(
        first["config"]["fit_frames"] + first["config"]["validation_frames"]
    ):
        image = quote(
            os.path.relpath(
                reports[0][0].parent / f"frame-{frame:06d}.jpg", output.parent
            )
        )
        chunks.append(
            f'<figure><img loading="lazy" src="{image}" alt="Three camera contact projections for frame {frame}"><figcaption>Frame {frame}</figcaption></figure>'
        )
    output.write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_review(args.audit, args.output)


if __name__ == "__main__":
    main()
