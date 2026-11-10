"""Image-only review of frozen hand depth-support boundaries."""

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real2sim2real.artifacts import digest, write_json
from .hand_depth_support import load_boundaries, signed_support_distance, support_region
from .interaction_io import Reconstruction, fresh_directory


def review(config):
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    boundaries, receipt = load_boundaries(
        config["boundaries"],
        source.frame_ids,
        source.times,
        config["blocked_frame_ids"],
    )
    probes = json.loads(Path(config["probes"]).read_text())
    out = fresh_directory(config["output"])
    rows, panels, hashes = [], [], {}
    for frame in config["frame_ids"]:
        found = np.flatnonzero(source.frame_ids == frame)
        if len(found) != 1:
            raise ValueError("Support review frame absent from capture")
        i = int(found[0])
        path = source.path(f"rgb/{i + source.start:06d}.png")
        hashes[str(path)] = digest(path)
        rgb = source.rgb(i)
        for margin in config["margins_px"]:
            image = rgb.copy()
            for h in range(2):
                mask = source.mask(i, config["mask_ids"][h])
                mask_path = source.path(
                    f"masks/{config['mask_ids'][h]}/{i + source.start:06d}.png"
                )
                hashes[str(mask_path)] = digest(mask_path)
                region = support_region(mask.shape, boundaries, i, h, margin)
                excluded = mask & ~region
                image[excluded] = (
                    image[excluded] * 0.35 + np.array([230, 40, 40]) * 0.65
                )
                row = dict(
                    frame_id=frame,
                    hand=h,
                    margin_px=margin,
                    boundary_available=bool(boundaries["available"][i, h]),
                    foreground_fraction_kept=float(
                        (mask & region).sum() / max(1, mask.sum())
                    ),
                    probes=[],
                )
                for probe in probes["points"]:
                    if probe["frame_id"] != frame or probe["hand"] != h:
                        continue
                    if probe["image_sha256"] != hashes[str(path)]:
                        raise ValueError("Boundary probe refers to a different image")
                    value = None
                    verdict = "unavailable"
                    if boundaries["available"][i, h]:
                        value = float(
                            signed_support_distance(
                                probe["xy_px"],
                                boundaries["wrist_px"][i, h],
                                boundaries["palm_direction"][i, h],
                                margin,
                            )
                        )
                        oriented = (
                            value if probe["expected"] == "retain_hand" else -value
                        )
                        verdict = (
                            "pass"
                            if oriented >= probe["uncertainty_px"]
                            else "uncertain"
                            if oriented >= -probe["uncertainty_px"]
                            else "fail"
                        )
                    row["probes"].append(
                        dict(probe, signed_distance_px=value, verdict=verdict)
                    )
                    cv2.drawMarker(
                        image,
                        tuple(probe["xy_px"]),
                        (50, 240, 90)
                        if probe["expected"] == "retain_hand"
                        else (255, 220, 40),
                        cv2.MARKER_CROSS,
                        12,
                        2,
                    )
                rows.append(row)
            filename = f"frame_{frame}_margin_{margin}.jpg"
            Image.fromarray(image).save(out / filename, quality=90)
            panels.append(
                f'<details><summary>Frame {frame}, margin {margin} px</summary><img src="{filename}"></details>'
            )
    summary = dict(
        schema="real2sim2real.hand-support-review/1",
        boundary_receipt=receipt,
        margins_px=config["margins_px"],
        reviewer=probes["reviewer"],
        human_verified=probes["human_verified"],
        result_by_margin={
            str(m): {
                v: sum(
                    p["verdict"] == v
                    for r in rows
                    if r["margin_px"] == m
                    for p in r["probes"]
                )
                for v in ["pass", "uncertain", "fail", "unavailable"]
            }
            for m in config["margins_px"]
        },
        source_hashes=hashes,
        probes_sha256=digest(config["probes"]),
        rows=rows,
        limitation="Sparse image probes assess this boundary proxy; they do not certify full anatomical segmentation or control readiness.",
        poses_or_depth_used=False,
    )
    write_json(out / "review.json", summary)
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Hand depth support</title><style>body{font:16px system-ui;background:#17202b;color:#eee;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}a{color:#8cf}</style><h1>Hand depth support review</h1><p>Red: excluded mask pixels beyond the observed wrist boundary. Green probes should retain hand pixels; yellow probes should exclude clear forearm pixels. Missing boundaries retain original support. No fitted hand poses or depth residuals were used.</p><a href="review.json">Full evidence</a><pre>'
        + html.escape(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in ("rows", "source_hashes")
                },
                indent=2,
            )
        )
        + "</pre>"
        + "".join(panels)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    review(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
