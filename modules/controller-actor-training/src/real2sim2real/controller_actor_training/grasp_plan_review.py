"""Static geometry review for admitted and rejected plans; no physics playback."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from real2sim2real.artifacts import digest, write_json


def review(config):
    import mujoco
    from .object_grasp_pilot import load_plan, load_scene, set_wrist_frame
    from .object_grasp_review import display_pieces, world_triangles, panel
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    plan = load_plan(config)
    output = Path(config["output"]) / config.get("plan_review_directory", "plan-review")
    output.mkdir(exist_ok=False)
    model, data, mesh = load_scene(config)
    idx = robot_indices(model, config["robot"]["root_body"])
    qid = model.jnt_qposadr[model.joint("task_object_free").id]
    pieces = display_pieces(model)
    mesh.apply_transform(plan["settling"]["settled_world_from_object"])
    center = mesh.bounds.mean(0)
    rows, cards, images = [], [], []
    ordered = sorted(
        plan["candidates"], key=lambda c: (len(c["rejected_for"]), c["score"])
    )
    for c in ordered:
        balance = c.get("pad_plan", {}).get("gravity_wrench", {})
        reason = html.escape(", ".join(c["rejected_for"]) or "geometric admission only")
        rows.append(
            f"<tr><td>{c['candidate_id']}</td><td>{c['accepted']}</td><td>{max(c['site_errors_m']) * 1000:.2f}</td><td>{reason}</td></tr>"
        )
        if len(cards) >= 6:
            continue
        set_wrist_frame(model, data, config, c["base_position_m"], c["base_rotation"])
        data.qpos[qid : qid + 7] = plan["settling"]["object_qpos"]
        panels = []
        for phase, key in [
            ("open", "q_open"),
            ("fit", "q_grasp"),
            ("closed", "q_closed"),
        ]:
            if key not in c:
                continue
            data.qpos[idx["q"]] = c[key]
            mujoco.mj_fwdPosition(model, data)
            triangles, colors = world_triangles(data, pieces)
            pair = Image.new("RGB", (1040, 500), "white")
            for j, (az, el) in enumerate([(40, 25), (135, 55)]):
                pair.paste(
                    panel(triangles, colors, az, el, center, 1600, depth_buffer=True),
                    (j * 520, 40),
                )
            ImageDraw.Draw(pair).text(
                (12, 12),
                f"Candidate {c['candidate_id']} | {phase} | Kinematic hypothesis only; no physics trial",
                fill="black",
            )
            path = output / f"candidate-{c['candidate_id']}-{phase}.jpg"
            pair.save(path, quality=88)
            images.append(dict(path=path.name, sha256=digest(path)))
            panels.append(
                f'<figure><img src="{path.name}" alt="{phase} geometric hypothesis"><figcaption>{phase}</figcaption></figure>'
            )
        cards.append(
            f"<section><h2>Candidate {c['candidate_id']}</h2><p>{reason}. Static balance: {balance.get('feasible', 'not checked')}.</p>{''.join(panels)}</section>"
        )
    body = '<!doctype html><meta charset="utf-8"><title>Grasp geometry review</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}img{max-width:100%}td,th{padding:.5em;border:1px solid #ccc}table{border-collapse:collapse}section{margin:3em 0}figure{margin:1em 0}</style>'
    body += "<h1>Grasp geometry review</h1><p><strong>Static kinematic hypotheses, including failures. These images are not successful grasp demonstrations or physics videos.</strong></p>"
    body += f"<p>Geometrically admitted: {plan['accepted']}/{len(plan['candidates'])}. Force assumptions and rejection reasons remain in the saved plan.</p>"
    body += (
        "<table><tr><th>Candidate</th><th>Admitted</th><th>Max fit error mm</th><th>Reasons</th></tr>"
        + "".join(rows)
        + "</table>"
        + "".join(cards)
    )
    (output / "review.html").write_text(body)
    write_json(
        output / "review.json",
        dict(
            physics_run=False,
            images=images,
            plan_sha256=digest(
                Path(config["output"])
                / config.get("planning_directory", "planning")
                / "plan.json"
            ),
            scene_receipt_sha256=plan["scene_receipt_sha256"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=config["deadlines_s"]["review"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    review(config)


if __name__ == "__main__":
    main()
