"""Read-only schematic and admission table for a saved bearing support search."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from real2sim2real.artifacts import digest, write_json


def schematic(sites, primary, proposed, center):
    """Top/side point diagrams, explicitly not a collision or evidence rendering."""
    sets = [
        (np.asarray(sites).reshape(-1, 3), "#93bda6"),
        (np.asarray(primary).reshape(-1, 3), "#1761b0"),
        (np.asarray(proposed).reshape(-1, 3), "#cf352f"),
        (np.asarray([center]), "#171717"),
    ]
    all_points = np.vstack([p for p, _ in sets])
    panels = []
    for slot, axes in enumerate(((0, 1), (1, 2))):
        lo, hi = all_points[:, axes].min(0), all_points[:, axes].max(0)
        size = np.maximum(hi - lo, 0.03)
        out = [
            f'<g transform="translate({slot * 500},0)">',
            f'<text x="30" y="25">{"Top: x/y" if slot == 0 else "Side: y/z"} (metres)</text>',
        ]
        for points, color in sets:
            for point in points:
                x, y = (
                    35 + 420 * (point[axes[0]] - lo[0]) / size[0],
                    320 - 260 * (point[axes[1]] - lo[1]) / size[1],
                )
                radius = 2 if color == "#93bda6" else 5
                out.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>'
                )
        out.append(
            f'<text x="30" y="350">Range {lo.round(3).tolist()} to {hi.round(3).tolist()}</text></g>'
        )
        panels.extend(out)
    return (
        '<svg role="img" aria-label="Contact site schematic" viewBox="0 0 1000 370">'
        + "".join(panels)
        + "</svg>"
    )


def render(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    source = plan["config"]["source_trial"]
    source_path = Path(source["path"])
    if digest(source_path) != source["sha256"]:
        raise ValueError("Source grasp changed")
    parent = json.loads(source_path.read_text())
    survey_path = path.parent / "surface-survey.json"
    survey = (
        json.loads(survey_path.read_text())
        if survey_path.exists()
        else {"sites": [], "sampled_positions": 0}
    )
    sites = [s["targets_m"][0] for s in survey["sites"]]
    hand = parent["candidate"]
    rows = []
    for c in plan["candidates"]:
        metric = lambda values, scale=1: f"{max(values) * scale:.3f}" if values else "—"
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in (
                    c["index"],
                    c["proposal"].get("name", ""),
                    metric(c.get("site_errors_m"), 1000),
                    f"{c['maximum_all_pose_normal_error_deg']:.3f}"
                    if "maximum_all_pose_normal_error_deg" in c
                    else metric(c.get("normal_errors_deg")),
                    f"{c['path']['max_penetration_m'] * 1000:.3f}"
                    if "path" in c
                    else "—",
                    c.get("patches", {}).get("samples_per_pad", "—"),
                    c.get("gravity_screens", {})
                    .get(str(parent["configuration"]["physics"]["friction"]), {})
                    .get("feasible", "unavailable"),
                    c.get("paired_finite_gravity_wrench", {}).get(
                        "feasible", "unavailable"
                    ),
                    ", ".join(c["rejected_for"])
                    or "Geometry pass; control not admitted",
                )
            )
            + "</tr>"
        )
    fitted = [c for c in plan["candidates"] if "site_errors_m" in c]
    selected = min(fitted, key=lambda c: max(c["site_errors_m"])) if fitted else None
    proposed = selected["targets_m"] if selected else []
    drawing = schematic(
        sites, hand["targets_m"], proposed, hand["pair"]["center_of_mass_m"]
    )
    (path.parent / "review.html").write_text(
        """<!doctype html><meta charset="utf-8">
<style>body{font:17px system-ui;max-width:1200px;margin:30px auto}td,th{border:1px solid #ddd;padding:8px}table{border-collapse:collapse;font-size:14px}svg{width:100%}pre{white-space:pre-wrap}</style>
<h1>Support bearing geometry and load-path audit</h1>
<p><strong>No new control or dynamics validation.</strong> Targets and static equilibrium are hypotheses. Source-mesh geometry does not independently verify visible contact patches.</p>
<p>Green: table-clear underside survey sites, when supplied. Blue: retained primary targets. Red: best position-fit support proposal. Black: source mesh centre of mass. These are point schematics, not collision renderings.</p>"""
        + drawing
        + f"<p>Optional survey: {survey['sampled_positions']} positions; {len(sites)} exterior five-ray stencils pass the geometric gap screen. Geometry candidates: {plan['geometry_candidates']}.</p>"
        + "<table><tr><th>ID</th><th>Proposal</th><th>Max seating error mm</th><th>Max normal error deg</th><th>Path penetration mm</th><th>Finite samples per pad</th><th>Point balance</th><th>Finite paired balance</th><th>Rejections</th></tr>"
        + "".join(rows)
        + '</table><p>Admission uses the pinned source position/penetration limits, ≤25° normal error, native joint limits and finite contacts on all five pads. Static force feasibility does not establish actuator feasibility or grasp retention. The primary hand is held at its source closed pose for this preflight; simultaneous acquisition and arm reachability remain untested.</p><p><a href="plan.json">Full plan and provenance</a> · {SURVEY_LINK}</p>'
    )
    review_path = path.parent / "review.html"
    review_path.write_text(
        review_path.read_text().replace(
            "{SURVEY_LINK}",
            '<a href="surface-survey.json">Surface survey</a>'
            if survey_path.exists()
            else "No separate survey supplied",
        )
    )
    write_json(
        path.parent / "review-receipt.json",
        dict(
            plan_sha256=digest(path),
            survey_sha256=digest(survey_path) if survey_path.exists() else None,
            review_sha256=digest(path.parent / "review.html"),
            implementation_sha256=digest(Path(__file__)),
            physics_rerun=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    render(parser.parse_args().plan)


if __name__ == "__main__":
    main()
