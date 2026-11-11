"""Static opening, cross-section and capture review for insertion diagnostics."""

import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest
from real2sim2real.egocentric.camera import transform
from real2sim2real.egocentric.interaction_io import Reconstruction


def write_review(audit_path, output, reconstruction=None):
    audit_path, output = Path(audit_path), Path(output)
    r = json.loads(audit_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    link = lambda p: quote(os.path.relpath(p, output.parent))
    chunks = [
        '<!doctype html><meta charset="utf-8"><title>Handle opening and insertion</title>',
        "<style>body{font:17px sans-serif;max-width:1100px;margin:30px auto;padding:15px;background:#17202b;color:#eee}td,th{border:1px solid #65758a;padding:7px}table{border-collapse:collapse}img{max-width:100%}a{color:#6ddcff}</style>",
        "<h1>Handle opening and insertion clearance</h1><p>Prepared mesh and finite table, fixed settled pose. These are idealized probes and rigid hook/support envelopes, not robot fingers or successful grasp rollouts. Hidden bearing surfaces remain unverified.</p>",
    ]
    for opening in r["openings"]:
        chunks.append(
            f"<p>Sampled opening area: {opening['area_m2'] * 1e6:.0f} mm²; inscribed diameter estimate: {opening['raster_inscribed_radius_m'] * 2000:.1f} mm. Grid radius allowance: {opening['raster_error_allowance_m'] * 1000:.2f} mm. Collision proxy blocks {opening['proxy_blocked_fraction'] * 100:.1f}% of source opening cells.</p>"
        )
    for name in ["source", "proxy"]:
        chunks.append(
            f'<img src="{link(audit_path.parent / (name + "-opening.png"))}" alt="{name} projected opening">'
        )
    chunks.append(
        "<h2>Path checks</h2><p>Lower bounds include 1-Lipschitz sampling allowances and the configured safety margin. Source geometry, collision proxy and table must all pass. An opening pass does not establish hooking or load bearing.</p><table><tr><th>Path</th><th>Probe diameter (mm)</th><th>Clearance lower bound (mm)</th><th>Limiting obstacle</th><th>Pass</th></tr>"
    )
    for path in r["paths"]:
        for variant in path["variants"]:
            worst = min(
                variant["obstacles"], key=lambda o: o["conservative_clearance_m"]
            )
            values = [
                path["label"],
                f"{variant['radius_m'] * 2000:.0f}",
                f"{worst['conservative_clearance_m'] * 1000:.2f}",
                worst["obstacle"],
                variant["accepted"],
            ]
            chunks.append(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
            )
    chunks.append(
        "</table><h2>Tabletop gap needed at hook targets</h2><table><tr><th>Target</th><th>Current gap (mm)</th><th>Diameter → minimum additional gap (mm)</th></tr>"
    )
    for gap in r["hook_gap_requirements"]:
        values = [
            gap["label"],
            f"{gap['current_gap_m'] * 1000:.2f}",
            ", ".join(
                f"{v['radius_m'] * 2000:.0f} → {v['minimum_additional_gap_m'] * 1000:.2f}"
                for v in gap["radii"]
            ),
        ]
        chunks.append(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
        )
    chunks.append(
        "</table><p>Gap bounds include the declared clearance margin, but omit robot shape, motion feasibility and geometry uncertainty. Creating the gap requires a separately planned and physically tested tilt/lift.</p>"
    )
    config_path = Path(r["config"]["source_config"])
    if digest(config_path) != r["config"]["source_config_sha256"]:
        raise ValueError("Review planning source changed")
    settings = json.loads(config_path.read_text())
    root = Path(settings["output"]) / "scene"
    if digest(root / "receipt.json") != r["scene_receipt_sha256"]:
        raise ValueError("Review scene changed")
    receipt = json.loads((root / "receipt.json").read_text())
    if digest(root / "object.obj") != receipt["assets"]["object.obj"]:
        raise ValueError("Review mesh changed")
    mesh = trimesh.load(root / "object.obj", force="mesh", process=False)
    mesh.apply_transform(r["world_from_object"])
    if r["openings"]:
        center = r["openings"][0]["center_xy_m"]
        section = mesh.section(plane_origin=[center[0], 0, 0], plane_normal=[1, 0, 0])
        canvas = Image.new("RGB", (1000, 420), "#f3f3f3")
        draw = ImageDraw.Draw(canvas)
        ymin, ymax = mesh.bounds[0, 1] - 0.01, mesh.bounds[1, 1] + 0.01
        scale = 900 / (ymax - ymin)

        def pixel(p):
            return (50 + (p[1] - ymin) * scale, 350 - p[2] * scale)

        table = trimesh.load(root / "table.obj", force="mesh", process=False)
        table_z = float(table.bounds[1, 2])
        top = pixel([0, 0, table_z])[1]
        draw.rectangle((0, top, 1000, 420), fill="#ab825a")
        if section is not None:
            for entity in section.entities:
                curve = entity.discrete(section.vertices)
                draw.line([pixel(p) for p in curve], fill="#195363", width=3)
        chosen = next(
            (
                p
                for p in r["paths"]
                if p["kind"] == "hook_envelope" and "yaw90" in p["label"]
            ),
            None,
        )
        if chosen:
            points = np.asarray(chosen["skeleton_m"]) + chosen["translation_path_m"][-1]
            width = max(1, round(2 * min(r["config"]["radii_m"]) * scale))
            draw.line([pixel(p) for p in points], fill="#d14c48", width=width)
        draw.text(
            (15, 10),
            "Metric side section: cyan object; brown table; red example hook envelope (rejected)",
            fill="black",
        )
        draw.text(
            (15, 30),
            "Equal spatial scale. This is geometry, not a physics simulation.",
            fill="black",
        )
        canvas.save(output.parent / "section.png")
        chunks.append(
            '<h2>Side section</h2><img src="section.png" alt="Object and hook clearance above the table">'
        )
    if reconstruction is not None and r["openings"]:
        source = Reconstruction(reconstruction)
        task = json.loads(Path(settings["task"]).read_text())
        if task["object"]["sha256"] != digest(source.path("mesh/metric.glb")) or any(
            digest(source.path(k)) not in task["sources"].values()
            for k in ["geometry.npz", "timeline.json"]
        ):
            raise ValueError("Review capture and task differ")
        opening = r["openings"][0]
        center = np.array(opening["display_center_task_m"])
        angles = np.linspace(0, 2 * np.pi, 65)
        radius = (
            opening["raster_inscribed_radius_m"] - opening["raster_error_allowance_m"]
        )
        ring = (
            center
            + np.c_[
                np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(len(angles))
            ]
        )
        local = transform(ring, np.linalg.inv(r["world_from_object"]))
        chunks.append(
            "<h2>Projection into capture</h2><p>Cyan is the model’s vertical opening corridor at its midheight. These overlays help visual review; they do not verify hidden hook-bearing surfaces.</p>"
        )
        for frame in np.linspace(0, source.count - 1, 4, dtype=int):
            p = transform(
                local,
                np.linalg.inv(source.world_from_camera[frame])
                @ source.world_from_object[frame],
            )
            uv = p @ source.k.T
            uv = uv[:, :2] / uv[:, 2:]
            image = Image.fromarray(source.rgb(frame))
            draw = ImageDraw.Draw(image)
            if (p[:, 2] > 0).all():
                draw.line([tuple(v) for v in uv], fill="#00ddff", width=3)
            path = output.parent / f"capture-{frame:06d}.jpg"
            image.save(path, quality=90)
            chunks.append(
                f'<figure><img src="{path.name}" alt="Model opening projection frame {frame}"><figcaption>Capture frame {frame}</figcaption></figure>'
            )
    chunks.append(f'<p><a href="{link(audit_path)}">Full audit JSON</a></p>')
    output.write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path)
    args = parser.parse_args()
    write_review(args.audit, args.output, args.reconstruction)


if __name__ == "__main__":
    main()
