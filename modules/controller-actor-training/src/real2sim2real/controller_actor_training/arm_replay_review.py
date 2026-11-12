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


def render(directory, *, follow_object=False):
    directory = Path(directory)
    stem = "rollout-follow" if follow_object else "rollout"
    if (directory / (stem + ".mp4")).exists():
        raise FileExistsError(
            "Saved review video exists; use --refresh-page to preserve immutable media"
        )
    import mujoco
    import imageio_ffmpeg

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
    stem = "rollout-follow" if follow_object else "rollout"
    video = directory / (stem + ".mp4")
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
            if follow_object:
                close_center = data.xpos[model.body("task_object").id].copy() + [
                    0,
                    0,
                    0.04,
                ]
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
            draw.text(
                (12, 25),
                "Finger control: "
                + report.get("finger_control_mode", "archived")
                + " | "
                + report.get("motion_condition", "demonstrated")
                + (
                    " | right camera follows object; left overview fixed"
                    if follow_object
                    else ""
                ),
                fill="black",
            )
            writer.send(np.asarray(frame))
    finally:
        writer.close()
    refresh_page(directory, follow_object=follow_object)


def motion_explanation(report):
    """Distinguish deliberately frozen robot goals from truncated playback."""
    records = report.get("records", [])
    if report.get("motion_condition") != "stationary_hold":
        return "<p>This replay uses the demonstrated-motion condition. Check the result for task acceptance.</p>"
    hold = next((r for r in records if r.get("phase") == "hold"), None)
    # Saved record timestamps are seconds in the simulation clock.
    start = hold.get("time_s") if hold else None
    end = records[-1].get("time_s") if records else None
    timing = (
        f" at {start:.2f} s; physics and playback continue through {end:.2f} s"
        if start is not None and end is not None
        else "; physics continues throughout the saved stationary test"
    )
    return (
        '<p role="note"><strong>Stationary diagnostic: the commanded lift deliberately stops'
        + timing
        + ".</strong> The object remains free to move or slip. This tests grasp retention; "
        + "it is not the full demonstrated trajectory. Return and finger release are disabled. "
        + "A stationary result does not establish demonstrated-motion success.</p>"
    )


def refresh_page(directory, *, follow_object=False):
    """Refresh report text using immutable saved video; do not rerender or simulate."""
    directory = Path(directory)
    report = json.loads((directory / "result.json").read_text())
    stem = "rollout-follow" if follow_object else "rollout"
    playback_name = "playback-follow" if follow_object else "playback"
    video = directory / (stem + ".mp4")
    receipt_path = directory / (
        "review-receipt-follow.json" if follow_object else "review-receipt.json"
    )
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["result_sha256"] != digest(directory / "result.json") or previous[
            "video_sha256"
        ] != digest(video):
            raise ValueError(
                "Saved video/result identity differs from the previous review"
            )
    playback = prepare_playback(video, directory / playback_name)
    media = video_html(
        stem + ".mp4",
        playback_name + "/video.webm",
        playback_name + "/preview.gif",
        embedded_webm=directory / playback_name / "video.webm",
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
        "finger_control_mode",
        "motion_condition",
        "controller_period_s",
        "physics_timestep_s",
        "finger_plan_admission",
        "stationary_admission",
        "residual_adapter",
        "bounded_grasp_correction",
        "closed_loop_grasp_feedback",
        "descent_pad_control",
        "continuous_protocol",
        "continuous_metrics",
    ):
        if key in report:
            metrics[key] = report[key]
    finger_description = (
        "live finger-pressure feedback with privileged simulator observations"
        if report.get("finger_control_mode") == "live_contact_force"
        else "recorded finger commands"
    )
    release = (report.get("continuous_metrics") or {}).get("release")
    phase_summary = ""
    if release:
        phase_summary = (
            "<p><strong>Complete controller: "
            + ("passed" if report["success"] else "rejected")
            + ". Finger release: "
            + ("passed" if release.get("accepted") else "rejected")
            + ".</strong></p>"
        )
    (directory / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}video{width:100%}pre{white-space:pre-wrap}</style><h1>Two-arm control transfer</h1><p>Saved simulated dynamics, not hardware execution. Both Cartesian wrist fixtures are removed. Torque-limited arm actuators and '
        + finger_description
        + " act on a free object. The straight mounts are hypotheses; hardware assembly and policy readiness are not established.</p>"
        + "<p>Motion condition: "
        + html.escape(report.get("motion_condition", "demonstrated"))
        + ".</p>"
        + motion_explanation(report)
        + phase_summary
        + media
        + '<p><a href="result.json">Full result and provenance</a></p><pre>'
        + html.escape(json.dumps(metrics, indent=2))
        + "</pre>"
    )
    write_json(
        directory
        / ("review-receipt-follow.json" if follow_object else "review-receipt.json"),
        dict(
            result_sha256=digest(directory / "result.json"),
            video_sha256=digest(video),
            playback=playback,
            implementation_sha256=digest(Path(__file__)),
            physics_rerun=False,
            right_camera_follows_object=follow_object,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--follow-object", action="store_true")
    parser.add_argument(
        "--refresh-page",
        action="store_true",
        help="Reuse saved video; update report text only",
    )
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
                *(["--follow-object"] if args.follow_object else []),
                *(["--refresh-page"] if args.refresh_page else []),
            ],
            timeout=600,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    action = refresh_page if args.refresh_page else render
    action(args.directory, follow_object=args.follow_object)


if __name__ == "__main__":
    main()
