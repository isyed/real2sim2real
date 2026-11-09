"""Stream rectified RGB and stereo depth into one calibrated image grid."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from real2sim2real.artifacts import write_json
from .capture import frames
from .camera import resize_intrinsic
from .stereo import Stereo


def prepare(capture, output, count, calibration, *, image_scale=0.5):
    output = Path(output)
    if not 1 < count <= len(capture["frame_ids"]) or not 0 < image_scale <= 1:
        raise ValueError("Invalid frame count or output scale")
    if not calibration["accepted"]:
        raise ValueError("Stereo calibration did not pass independent image validation")
    target = capture["streams"][capture["target_stream"]]
    left, right = [capture["streams"][n] for n in capture["stereo_pair"]]
    model = Stereo(
        left, right, target, right_from_left=calibration["candidate_right_from_left"]
    )
    size = tuple(round(v * image_scale) for v in target["size"])
    k = resize_intrinsic(target["intrinsics"], target["size"], size)
    maps = cv2.initUndistortRectifyMap(
        np.asarray(target["intrinsics"], float),
        np.asarray(target["distortion"], float),
        np.eye(3),
        k,
        size,
        cv2.CV_32FC1,
    )
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(exist_ok=True)
    rows = []
    for i, (rgb, li, ri) in enumerate(
        zip(
            frames(target, count),
            frames(left, count),
            frames(right, count),
            strict=True,
        )
    ):
        rgb = cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR)
        depth, facts = model.depth(li, ri, k, size)
        Image.fromarray(rgb).save(output / "rgb" / f"{i:06d}.png")
        np.save(output / "depth" / f"{i:06d}.npy", depth)
        rows.append(
            dict(
                frame_id=capture["frame_ids"][i],
                target_valid_fraction=float((depth > 0).mean()),
                **facts,
            )
        )
        print(f"prepared frame {i + 1}/{count}", flush=True)
    if len(rows) != count:
        raise ValueError("Capture streams ended before the declared timeline")
    write_json(
        output / "camera.json",
        dict(
            fx=k[0, 0],
            fy=k[1, 1],
            cx=k[0, 2],
            cy=k[1, 2],
            width=size[0],
            height=size[1],
        ),
    )
    write_json(
        output / "timeline.json",
        dict(
            frame_ids=capture["frame_ids"][:count],
            timestamps_s=capture["timestamps_s"][:count],
            intrinsics=k.tolist(),
            image_size=list(size),
            capture_sha256=capture["manifest_sha256"],
            depth_source="calibrated_stereo_sgbm",
            frames=count,
        ),
    )
    write_json(output / "depth_quality.json", rows)
    return rows
