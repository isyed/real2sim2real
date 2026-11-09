"""CPU inspection artifacts for masks and metric depth; no acceptance by appearance."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from real2sim2real.artifacts import write_json


def review(output, frame_ids, *, object_id=1, occluder_ids=()):
    output = Path(output)
    panels = []
    rows = []
    for frame in frame_ids:
        rgb = np.asarray(
            Image.open(output / "rgb" / f"{frame:06d}.png").convert("RGB")
        ).copy()
        depth = np.load(output / "depth" / f"{frame:06d}.npy")
        object_mask = (
            np.asarray(
                Image.open(output / "masks" / str(object_id) / f"{frame:06d}.png")
            )
            > 0
        )
        measured = object_mask & (depth > 0)
        row = dict(
            frame=frame,
            object_pixels=int(object_mask.sum()),
            object_depth_coverage=float(measured.sum() / max(object_mask.sum(), 1)),
            median_depth_m=float(np.median(depth[measured]))
            if measured.any()
            else None,
        )
        rows.append(row)
        palette = [[255, 80, 80], [80, 100, 255], [240, 180, 40]]
        colors = [(object_id, [0, 255, 100])] + [
            (obj, palette[i % len(palette)]) for i, obj in enumerate(occluder_ids)
        ]
        for obj, color in colors:
            mask = (
                np.asarray(Image.open(output / "masks" / str(obj) / f"{frame:06d}.png"))
                > 0
            )
            rgb[mask] = (rgb[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        image = Image.fromarray(rgb).resize((507, 380))
        ImageDraw.Draw(image).text(
            (8, 8),
            f"frame {frame}; depth coverage {row['object_depth_coverage']:.1%}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        panels.append(image)
    sheet = Image.new("RGB", (507 * len(panels), 380))
    for i, image in enumerate(panels):
        sheet.paste(image, (i * 507, 0))
    sheet.save(output / "mask_review.jpg")
    write_json(output / "mask_review.json", rows)
    return rows
