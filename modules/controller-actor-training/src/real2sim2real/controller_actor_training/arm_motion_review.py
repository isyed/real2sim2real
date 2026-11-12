"""Synchronized browser comparison of two pinned full-motion review videos.

Different audited grasps/paths are allowed and explicitly reported; this is a
visual comparison, not a controlled single-variable experiment or promotion.
"""

import argparse
import html
import json
import os
from pathlib import Path

from real2sim2real.artifacts import digest, write_json
from real2sim2real.review_comparison import side_by_side
from .paired_acquisition_audit import pinned_json


def build(config):
    if len(config["runs"]) != 2:
        raise ValueError("Exactly two pinned motion reviews required")
    out = Path(config["output"])
    rows, sections, clocks, videos = [], [], [], []
    for entry in config["runs"]:
        result = pinned_json(entry["result"])
        path = Path(entry["result"]["path"])
        receipt = json.loads(path.with_name("review-receipt-follow.json").read_text())
        video = path.with_name("rollout-follow.mp4")
        videos.append(video)
        if (
            receipt["result_sha256"] != digest(path)
            or receipt["video_sha256"] != digest(video)
            or result.get("motion_condition") != "demonstrated"
            or not result.get("demonstrated_motion_executed")
        ):
            raise ValueError("Pinned rendered full-motion physics required")
        protocol = result["continuous_protocol"]
        clocks.append(
            (
                protocol["frame_ids"],
                protocol["capture_duration_s"],
                [(r["time_s"], r["phase"]) for r in result["records"]],
            )
        )
        webm = path.parent / "playback-follow/video.webm"
        media = receipt["playback"]["files"]["webm"]
        if digest(webm) != media["sha256"]:
            raise ValueError("Browser playback changed")
        href = lambda p: html.escape(os.path.relpath(p, out), quote=True)
        sections.append(
            "<section><h2>" + html.escape(entry["label"]) + "</h2>"
            '<video muted controls preload="metadata"><source src="'
            + href(webm)
            + '" type="video/webm"><source src="'
            + href(video)
            + '" type="video/mp4"></video><p><a href="'
            + href(video)
            + '">Open MP4</a> · <a href="'
            + href(path.with_name("review.html"))
            + '">Full review</a></p><p>Rejected: '
            + html.escape(", ".join(result["rejected_for"]) or "none")
            + "</p></section>"
        )
        rows.append(
            dict(
                label=entry["label"],
                result=entry["result"],
                video_sha256=digest(video),
                audit_sha256=result["config"]["audit_sha256"],
                success=result["success"],
                rejected_for=result["rejected_for"],
                release=result["continuous_metrics"]["release"],
                primary_drift_mm=result["hold_slip_m"] * 1000,
                support_drift_mm=result["support_wrist_drift_m"] * 1000,
                minimum_clearance_mm=result["hold_minimum_clearance_m"] * 1000,
                peak_reference_position_error_mm=max(
                    (
                        r["reference_position_error_m"] * 1000
                        for r in result["records"]
                        if "reference_position_error_m" in r
                    ),
                    default=None,
                ),
                peak_reference_rotation_error_deg=max(
                    (
                        r["reference_rotation_error_deg"]
                        for r in result["records"]
                        if "reference_rotation_error_deg" in r
                    ),
                    default=None,
                ),
                maximum_robot_penetration_mm=result["max_robot_penetration_m"] * 1000,
                maximum_arm_penetration_mm=result["max_arm_penetration_m"] * 1000,
                bounded_grasp_correction=result.get("bounded_grasp_correction"),
                closed_loop_grasp_feedback=result.get("closed_loop_grasp_feedback"),
                descent_pad_control=result.get("descent_pad_control"),
                table_load_transfer=result.get("table_load_transfer"),
            )
        )
    if clocks[0] != clocks[1]:
        raise ValueError(
            "Synchronized review requires equal native clocks and phase samples"
        )
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="arm-motion-visual-comparison/1",
        rows=rows,
        matched_clock=True,
        audited_paths_equal=rows[0]["audit_sha256"] == rows[1]["audit_sha256"],
        command_corrections_equal=all(
            rows[0][k] == rows[1][k]
            for k in (
                "bounded_grasp_correction",
                "closed_loop_grasp_feedback",
                "descent_pad_control",
                "table_load_transfer",
            )
        ),
        controlled_single_variable_experiment=False,
        physics_rerun=False,
        implementation_sha256=digest(Path(__file__)),
    )
    combined_link = ""
    if config.get("combined_video", False):
        report["combined_video"] = side_by_side(
            videos, [row["label"] for row in rows], out / "comparison.mp4"
        )
        combined_link = '<p><a href="comparison.mp4">Open or download one synchronized side-by-side MP4</a>.</p>'
    write_json(out / "comparison.json", report)
    metric_rows = []
    for key, label in (
        ("primary_drift_mm", "Primary fixed-anchor drift (mm)"),
        ("support_drift_mm", "Support fixed-anchor drift (mm)"),
        ("minimum_clearance_mm", "Suspended minimum clearance (mm)"),
        (
            "peak_reference_position_error_mm",
            "Peak object-reference position error (mm)",
        ),
        (
            "peak_reference_rotation_error_deg",
            "Peak object-reference rotation error (degrees)",
        ),
        ("maximum_robot_penetration_mm", "Maximum robot penetration (mm)"),
        ("maximum_arm_penetration_mm", "Maximum arm penetration (mm)"),
    ):
        values = ["—" if row[key] is None else f"{row[key]:.3f}" for row in rows]
        metric_rows.append(
            "<tr><th>"
            + label
            + "</th>"
            + "".join("<td>" + v + "</td>" for v in values)
            + "</tr>"
        )
    metric_rows.append(
        "<tr><th>Supported finger release</th>"
        + "".join(
            "<td>" + ("Pass" if row["release"]["accepted"] else "Reject") + "</td>"
            for row in rows
        )
        + "</tr>"
    )
    table = (
        "<table><thead><tr><th>Observed metric</th>"
        + "".join("<th>" + html.escape(row["label"]) + "</th>" for row in rows)
        + "</tr></thead><tbody>"
        + "".join(metric_rows)
        + "</tbody></table>"
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Full-motion grasp comparison</title>'
        "<style>body{font:17px system-ui;margin:24px;background:#f7f8fa;color:#17202b}"
        "main{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%}"
        "button{padding:10px;margin:6px}section{min-width:0}pre{white-space:pre-wrap}"
        "table{border-collapse:collapse;margin:20px 0}th,td{padding:8px;border:1px solid #ccd3dc;text-align:left}"
        "@media(max-width:800px){main{grid-template-columns:1fr}}</style>"
        "<h1>Full continuous motion: acquisition → lift → return → release</h1>"
        "<p>Saved simulation; both attempts retain their original acceptance status. "
        + html.escape(
            config.get(
                "description",
                "These may use different grasp/path configurations. "
                "This is a visual comparison, not an isolated test of closure order.",
            )
        )
        + " No diagnostic hold or time warp is added.</p>"
        '<button id="play">Play both</button><button id="pause">Pause both</button>'
        '<button id="restart">Restart both</button><span id="status"></span><main>'
        + "".join(sections)
        + "</main>"
        + combined_link
        + table
        + "<p>Use the left video to seek; the right follows. "
        "Object-following camera views are rendered independently. "
        '<a href="comparison.json">Metrics and provenance</a>.</p>'
        '<script>const [a,b]=document.querySelectorAll("video");'
        'const status=document.getElementById("status");'
        "async function play(){b.currentTime=a.currentTime;try{await Promise.all([a.play(),b.play()]);"
        'status.textContent=""}catch(e){status.textContent="Use the individual video controls or MP4 links."}}'
        'document.getElementById("play").onclick=play;'
        'document.getElementById("pause").onclick=()=>{a.pause();b.pause()};'
        'document.getElementById("restart").onclick=()=>{a.currentTime=0;b.currentTime=0;play()};'
        'a.addEventListener("seeking",()=>{b.currentTime=a.currentTime});'
        'a.addEventListener("timeupdate",()=>{if(Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime});'
        'a.addEventListener("pause",()=>b.pause());'
        'a.addEventListener("play",()=>{b.currentTime=a.currentTime;b.play().catch(()=>{})});'
        'a.addEventListener("ratechange",()=>{b.playbackRate=a.playbackRate});</script>'
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()))
