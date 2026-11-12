"""CPU-only saved-state review for free-object fixture diagnostics."""

import argparse
import html
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from real2sim2real.artifacts import digest, write_json


def render(directory):
    import mujoco
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from real2sim2real.review_media import prepare, player
    from .object_grasp_review import display_pieces, world_triangles, panel
    from .object_grasp_pilot import verify_scene_assets

    directory = Path(directory)
    report = json.loads((directory / "result.json").read_text())
    if report["schema"] != "fixture-hold/1" or not report["physics_run"]:
        raise ValueError("Require saved fixture physics")
    for name, key in [("states.npz", "states_sha256"), ("scene.xml", "scene_sha256")]:
        if digest(directory / name) != report[key]:
            raise ValueError("Saved fixture input changed")
    root, _ = verify_scene_assets(report["configuration"])
    tree = ET.fromstring((directory / "scene.xml").read_text())
    # mj_saveLastXML retains source-relative asset names. Resolve only filenames;
    # model geometry/dynamics and the saved XML remain unchanged.
    compiler = tree.find("compiler")
    for name in ("meshdir", "texturedir"):
        compiler.set(name, str((root / compiler.get(name, "")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    with np.load(directory / "states.npz") as saved:
        q, times = saved["qpos"].copy(), saved["timestamps_s"].copy()
    if q.shape != (len(times), model.nq) or len(times) != len(report["records"]):
        raise ValueError("State/sample correspondence changed")
    video = directory / "rollout.mp4"
    if video.exists():
        raise FileExistsError("Preserve existing review video")
    pieces = display_pieces(model)
    step = max(1, round(len(times) / 180))
    fps = 1 / (float(np.median(np.diff(times))) * step)
    data.qpos[:] = q[0]
    mujoco.mj_fwdPosition(model, data)
    center = data.xpos[model.body("task_object").id].copy() + [0, 0, 0.05]
    writer = imageio_ffmpeg.write_frames(
        str(video),
        (1040, 520),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        output_params=[
            "-threads",
            "1",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
        ],
    )
    writer.send(None)
    try:
        for i in range(0, len(times), step):
            data.qpos[:] = q[i]
            mujoco.mj_fwdPosition(model, data)
            triangles, colors = world_triangles(data, pieces)
            frame = Image.new("RGB", (1040, 520), "white")
            for x, azimuth in [(0, -60), (520, 30)]:
                frame.paste(panel(triangles, colors, azimuth, 20, center, 900), (x, 40))
            draw = ImageDraw.Draw(frame)
            draw.text(
                (10, 8),
                f"Two articulated hands on wrist fixtures | saved PHYSICS | {times[i]:.2f}s | {report['records'][i]['phase']}",
                fill="black",
            )
            draw.text(
                (10, 25),
                report["config"]["grasp"]
                + " | "
                + report["config"]["condition"]
                + " | hold accepted: "
                + str(report["success"]),
                fill="black",
            )
            draw.text(
                (10, 503),
                "Diagnostic pause is intentional. Free object; no arms, object actuation or release test.",
                fill="black",
            )
            writer.send(np.asarray(frame))
    finally:
        writer.close()
    prepare(video, directory / "playback")
    media = player(
        "rollout.mp4",
        "playback/video.webm",
        "playback/preview.gif",
        embedded_webm=directory / "playback/video.webm",
    )
    metrics = {
        k: report[k]
        for k in (
            "success",
            "rejected_for",
            "metrics",
            "max_robot_penetration_m",
            "simulation_final_time_s",
            "simulation_requested_duration_s",
        )
    }
    (directory / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Fixture hold diagnostic</title><h1>Fixture hold diagnostic</h1><p>Saved simulation with torque-limited hands on Cartesian wrists. No physical arms or hardware. Commanded motion deliberately stops for the complete hold; playback and physics continue. This is not the full demonstration or a release test.</p>'
        + media
        + "<pre>"
        + html.escape(json.dumps(metrics, indent=2))
        + "</pre>"
    )
    write_json(
        directory / "review-receipt.json",
        dict(
            result_sha256=digest(directory / "result.json"),
            video_sha256=digest(video),
            states_sha256=report["states_sha256"],
            renderer_sha256=digest(Path(__file__)),
            last_saved_time_s=float(times[-1]),
            rendered_frames=len(range(0, len(times), step)),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if a.worker:
        render(a.directory)
    else:
        from real2sim2real.bounded_process import run

        run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--directory",
                str(a.directory.resolve()),
                "--worker",
            ],
            timeout=600,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    main()
