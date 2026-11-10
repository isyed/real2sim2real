"""CPU video review of saved arm dynamics; never recompute the rollout."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from real2sim2real.artifacts import digest, write_json
from real2sim2real.review_media import prepare as prepare_playback, player as video_html
from .object_grasp_review import display_pieces, world_triangles, panel


def render(directory):
    import mujoco
    import imageio_ffmpeg

    directory = Path(directory)
    report = json.loads((directory / "result.json").read_text())
    for filename, key in [
        ("states.npz", "states_sha256"),
        ("scene.xml", "scene_sha256"),
    ]:
        if digest(directory / filename) != report[key]:
            raise ValueError("Saved review input changed")
    with np.load(directory / "states.npz") as f:
        states, times = f["qpos"].copy(), f["timestamps_s"].copy()
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    # UR imports have collision geometry only. Display it as such; do not edit
    # the saved model or change any collision used by the experiment.
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name or ""
        if (
            name.startswith(("L__", "R__"))
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            model.geom_group[g] = 1
    pieces = display_pieces(model)
    data.qpos[:] = states[0]
    mujoco.mj_fwdPosition(model, data)
    vertices = world_triangles(data, pieces)[0].reshape(-1, 3)
    center = (vertices.min(0) + vertices.max(0)) / 2
    close_center = data.xpos[model.body("task_object").id].copy() + [0, 0, 0.08]
    scale = 420 / max(np.ptp(vertices, axis=0).max(), 0.1)
    step = max(1, round(len(times) / 160))
    fps = 1 / (float(np.median(np.diff(times))) * step)
    video = directory / "rollout.mp4"
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
        for i in range(0, len(states), step):
            data.qpos[:] = states[i]
            mujoco.mj_fwdPosition(model, data)
            triangles, colors = world_triangles(data, pieces)
            frame = Image.new("RGB", (1040, 520), "white")
            frame.paste(panel(triangles, colors, -60, 25, center, scale), (0, 40))
            frame.paste(panel(triangles, colors, 20, 15, close_center, 900), (520, 40))
            draw = ImageDraw.Draw(frame)
            draw.text(
                (12, 10),
                f"Two articulated arms with grasp hands | saved PHYSICS | {times[i]:.2f}s | {report['records'][i]['phase']}",
                fill="black",
            )
            draw.text(
                (12, 505),
                "Arm collision meshes shown. Mounts are simulation assumptions. Task accepted: "
                + str(report["success"]),
                fill="black",
            )
            writer.send(np.asarray(frame))
    finally:
        writer.close()
    playback = prepare_playback(video, directory / "playback")
    media = video_html(
        "rollout.mp4",
        "playback/video.webm",
        "playback/preview.gif",
        embedded_webm=directory / "playback/video.webm",
    )
    fields = [
        "success",
        "rejected_for",
        "hold_minimum_clearance_m",
        "hold_slip_m",
        "support_wrist_drift_m",
        "hold_opposed_contact_fraction",
        "support_load_fraction",
        "max_robot_penetration_m",
        "max_arm_penetration_m",
        "maximum_joint_tracking_error_rad",
    ]
    metrics = {key: report[key] for key in fields}
    for key in (
        "control_mode",
        "wrist_tracking_by_phase",
        "commanded_wrist_tracking_by_phase",
        "object_feedback",
        "feedback_observation_source",
        "contact_exclusions",
    ):
        if key in report:
            metrics[key] = report[key]
    (directory / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}video{width:100%}pre{white-space:pre-wrap}</style><h1>Two-arm control transfer</h1><p>Saved simulated dynamics, not hardware execution. Both Cartesian wrist fixtures are removed. Torque-limited arm actuators and recorded finger commands act on a free object. The straight mounts are hypotheses; hardware assembly and policy readiness are not established.</p>'
        + media
        + '<p><a href="result.json">Full result and provenance</a></p><pre>'
        + html.escape(json.dumps(metrics, indent=2))
        + "</pre>"
    )
    write_json(
        directory / "review-receipt.json",
        dict(
            result_sha256=digest(directory / "result.json"),
            video_sha256=digest(video),
            playback=playback,
            implementation_sha256=digest(Path(__file__)),
            physics_rerun=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--directory",
                str(args.directory.resolve()),
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
    render(args.directory)


if __name__ == "__main__":
    main()
