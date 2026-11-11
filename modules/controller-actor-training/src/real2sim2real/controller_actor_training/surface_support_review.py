"""Static review of surface discovery, held-out evidence and force screening."""

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_review(arrangement_path, output):
    arrangement = json.loads(Path(arrangement_path).read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        '<!doctype html><meta charset="utf-8"><title>Surface support search</title>',
        "<style>body{font:17px sans-serif;max-width:1100px;margin:30px auto;padding:15px;background:#17202b;color:#eee}td,th{padding:8px;border:1px solid #718091}table{border-collapse:collapse}img{width:100%}a{color:#6ddcff}</style>",
        "<h1>Observation-first support search</h1>",
        "<p>Surface hypotheses are sampled before robot fitting. Discovery alone freezes each shortlist; separate frames check its depth consistency. A supported point does not certify surface normals, contact area or robot access.</p>",
        f"<p>Supported sites across surveys: <strong>{len(arrangement['sites'])}</strong>. Static gravity balance: <strong>{arrangement['gravity_wrench']['feasible']}</strong>.</p>",
        f"<p>Optimistic upward-force bound: <strong>{arrangement['optimistic_upward_force_bound_n']:.3f} N</strong>; object weight: <strong>{arrangement['object_weight_n']:.3f} N</strong>, assuming {arrangement['per_site_normal_force_limit_n']:.1f} N normal force per site. This bound ignores lateral force and torque balance and uses reconstructed normals.</p>",
        "<p>No robot hands were fitted and no grasp/lift simulation was run. Physics assumptions and original rim plans are unchanged.</p>",
    ]
    for index, record in enumerate(arrangement["config"]["surveys"]):
        path = Path(record["path"])
        survey = json.loads(path.read_text())
        counts = Counter(s["discovery"]["evidence"] for s in survey["sites"])
        chunks.append(
            f"<h2>Survey {index}: {escape(survey['config'].get('sampling_mode', 'whole_surface'))}</h2><p>{len(survey['sites'])} sampled sites. Discovery: {escape(str(dict(counts)))}. Validation-supported sites: {escape(str(survey['supported_site_ids']))}.</p>"
        )
        chunks.append(
            "<table><tr><th>Site</th><th>Discovery matches</th><th>Validation matches</th><th>Validation median/p90 error (mm)</th><th>Evidence</th><th>Max upward N per normal N</th></tr>"
        )
        for site in survey["sites"]:
            if not site["shortlisted"]:
                continue
            s = site["validation"]
            fmt = lambda x: "missing" if x is None else f"{1000 * x:.2f}"
            values = [
                site["site_id"],
                site["discovery"]["observed_frames"],
                s["observed_frames"],
                f"{fmt(s['median_abs_depth_error_m'])} / {fmt(s['p90_abs_depth_error_m'])}",
                s["evidence"],
                f"{site['upward_force_per_unit_normal']:.4f}",
            ]
            chunks.append(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
            )
        chunks.append(
            "</table><p>Cyan: passes point-depth validation; orange: not admitted. Visibility depends on frame. These static overlays show locations, not a successful grasp.</p>"
        )
        for image in sorted(path.parent.glob("frame-*.jpg")):
            link = quote(os.path.relpath(image, output.parent))
            chunks.append(
                f'<figure><img loading="lazy" src="{link}" alt="{escape(image.stem)}"><figcaption>{escape(image.stem)}</figcaption></figure>'
            )
        link = quote(os.path.relpath(path, output.parent))
        chunks.append(f'<p><a href="{link}">Full survey</a></p>')
    output.write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrangement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_review(args.arrangement, args.output)


if __name__ == "__main__":
    main()
