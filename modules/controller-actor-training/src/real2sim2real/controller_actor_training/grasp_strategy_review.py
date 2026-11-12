"""CPU review of fresh strategy screens; static fits are never labelled physics."""

import argparse
import html
import json
import os
from pathlib import Path

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json
from .grasp_strategy_feedback import load_run


def build(config):
    import mujoco
    from PIL import Image, ImageDraw
    from .object_grasp_pilot import load_scene, set_root_frame
    from .object_grasp_review import display_pieces, world_triangles, panel, render
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    sections, rows, identities = [], [], []
    for i, entry in enumerate(config["runs"]):
        report, plan = load_run(entry)
        directory = Path(entry["evaluation"]["path"]).parent
        cat = pinned_json(report["config"]["catalogue"])
        identities.append(
            (
                cat["scene_config"]["sha256"],
                cat["scene_receipt_sha256"],
                cat["settling_sha256"],
                report["config"]["catalogue"]["sha256"],
            )
        )
        if identities[-1] != identities[0]:
            raise ValueError(
                "Matched strategy review requires one frozen scene, catalogue and protocol"
            )
        base = pinned_json(cat["scene_config"])
        label = html.escape(entry["label"])
        row = dict(
            label=entry["label"],
            evaluation_sha256=entry["evaluation"]["sha256"],
            geometric_pass=report["geometric_pass"],
            physics_executed=report["hold"]["executed"],
            simulated_hold_pass=report["hold"]["simulated_hold_pass"],
            rejected_for=report["rejected_for"],
            maximum_target_error_mm=None
            if not all("maximum_target_error_m" in h for h in plan["hands"])
            else max(h["maximum_target_error_m"] for h in plan["hands"]) * 1000,
            maximum_normal_error_deg=None
            if not all("maximum_normal_error_deg" in h for h in plan["hands"])
            else max(h["maximum_normal_error_deg"] for h in plan["hands"]),
            maximum_path_penetration_mm=None
            if plan["paired_path"] is None
            else plan["paired_path"]["max_penetration_m"] * 1000,
            samples_per_pad=[
                h.get("patches", {}).get("samples_per_pad") for h in plan["hands"]
            ],
            gravity_wrench_feasible=plan["gravity_wrench"]["feasible"],
            contact_observation_admitted=False,
        )
        rows.append(row)
        section = (
            "<section><h2>"
            + label
            + "</h2><p>"
            + html.escape(
                ", ".join(report["rejected_for"]) or "Geometric preflight passed"
            )
            + "</p>"
        )
        if all("joint_names" in h for h in plan["hands"]):
            model, data, mesh = load_scene(base)
            mesh.apply_transform(plan["settling"]["settled_world_from_object"])
            objq = model.joint("task_object_free").qposadr[0]
            data.qpos[objq : objq + 7] = plan["settling"]["object_qpos"]
            indices = []
            for hand in plan["hands"]:
                set_root_frame(
                    model,
                    data,
                    hand["root_body"],
                    hand["fixture_prefix"],
                    hand["base_position_m"],
                    hand["base_rotation"],
                    preserve_state=True,
                )
                indices.append(robot_indices(model, hand["root_body"]))
            pieces = display_pieces(model)
            center = mesh.bounds.mean(0)
            for phase in ("open", "closed"):
                for h, idx in zip(plan["hands"], indices):
                    data.qpos[idx["q"]] = h["q_" + phase]
                mujoco.mj_fwdPosition(model, data)
                triangles, colors = world_triangles(data, pieces)
                frame = Image.new("RGB", (1040, 520), "white")
                for j, angles in enumerate(((35, 30), (125, 65))):
                    frame.paste(
                        panel(
                            triangles, colors, *angles, center, 1450, depth_buffer=True
                        ),
                        (520 * j, 40),
                    )
                ImageDraw.Draw(frame).text(
                    (12, 12),
                    f"{entry['label']} | {phase} | Static fitted hypothesis; no loaded motion",
                    fill="black",
                )
                image = output / f"candidate-{i}-{phase}.jpg"
                frame.save(image, quality=90)
                section += (
                    '<img src="' + image.name + '" alt="Static ' + phase + ' fit">'
                )
        if report["hold"]["executed"]:
            physics = directory / "hold"
            saved = json.loads((physics / "result.json").read_text())
            render(
                saved["configuration"], physics, output, stem=f"candidate-{i}-physics"
            )
            section += f'<p><a href="candidate-{i}-physics.mp4">Saved hold physics video</a></p>'
        else:
            section += "<p>No hold video: rejected before physics. Images above are static geometry.</p>"
        section += (
            '<p><a href="'
            + html.escape(os.path.relpath(directory / "plan.json", output), quote=True)
            + '">Full fit and collision diagnostics</a></p></section>'
        )
        sections.append(section)
        load_run(entry)
    body = '<!doctype html><meta charset="utf-8"><title>Paired grasp strategy comparison</title><style>body{font:17px system-ui;max-width:1150px;margin:30px auto;padding:16px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}section{margin:40px 0}</style>'
    body += "<h1>Paired grasp strategies: acquisition and hold</h1><p>Conditional simulation study on reconstructed geometry. Observation admission, arm feasibility and full demonstrated motion remain unverified. All methods use the same prepared scene, catalogue, fitting budget and criteria.</p>"
    body += "<table><tr><th>Proposal</th><th>Geometry</th><th>Max fit error mm</th><th>Path penetration mm</th><th>Gravity balance</th><th>Hold</th></tr>"
    for row in rows:
        pen = row["maximum_path_penetration_mm"]
        fit = row["maximum_target_error_mm"]
        body += (
            "<tr><td>"
            + html.escape(row["label"])
            + f"</td><td>{row['geometric_pass']}</td><td>{'unavailable' if fit is None else f'{fit:.2f}'}</td><td>{'unavailable' if pen is None else f'{pen:.2f}'}</td><td>{row['gravity_wrench_feasible']}</td><td>"
            + (
                "Pass"
                if row["simulated_hold_pass"]
                else "Reject"
                if row["physics_executed"]
                else "Not run"
            )
            + "</td></tr>"
        )
    body += (
        "</table>"
        + "".join(sections)
        + '<p><a href="comparison.json">Metrics and provenance</a></p>'
    )
    (output / "review.html").write_text(body)
    write_json(
        output / "comparison.json",
        {
            "schema": "grasp-strategy-comparison/1",
            "config": config,
            "rows": rows,
            "control_ready": False,
            "implementation_sha256": digest(Path(__file__)),
        },
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()))
