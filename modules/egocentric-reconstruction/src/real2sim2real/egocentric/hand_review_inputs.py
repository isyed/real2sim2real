"""Held-out visual review inputs, separate from model-worker mounts."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real2sim2real.artifacts import digest, write_json
from .interaction_io import Reconstruction, fresh_directory


def prepare_review_inputs(config):
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    requested = config.get("review_frame_ids", source.frame_ids[::10].tolist())
    selected = np.searchsorted(source.frame_ids, requested)
    if (selected >= source.count).any() or not np.array_equal(
        source.frame_ids[selected], requested
    ):
        raise ValueError("Review frame IDs must belong to the experiment")
    out = fresh_directory(Path(config["output"]) / "evaluation_inputs")
    views = {
        "target": dict(
            intrinsics=source.k.tolist(), view_from_target=np.eye(4).tolist(), images={}
        )
    }
    for i in selected:
        path = out / f"target_{source.frame_ids[i]:06d}.png"
        Image.fromarray(source.rgb(i)).save(path)
        views["target"]["images"][str(source.frame_ids[i])] = dict(
            path=path.name, sha256=digest(path)
        )
    if config.get("capture"):
        from .capture import load, frames

        capture = load(config["capture"])
        if capture["manifest_sha256"] != source.timeline["capture_sha256"]:
            raise ValueError("Secondary streams must match the original capture")
        indices = selected + source.start
        if not np.allclose(
            np.asarray(capture["timestamps_s"])[indices],
            source.times[selected],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Secondary capture timestamps do not agree")
        target = capture["streams"][capture["target_stream"]]
        left_name, right_name = capture["stereo_pair"]
        left_from_target = np.linalg.inv(
            capture["streams"][left_name]["rig_from_camera"]
        ) @ np.asarray(target["rig_from_camera"])
        calibration = json.loads(source.path("calibration.json").read_text())
        if not calibration["accepted"]:
            raise ValueError("Secondary projections require accepted calibration")
        transforms = {
            left_name: left_from_target,
            right_name: np.asarray(calibration["candidate_right_from_left"])
            @ left_from_target,
        }
        for name in capture["stereo_pair"]:
            if name == capture["target_stream"]:
                continue
            stream = capture["streams"][name]
            k = np.asarray(stream["intrinsics"], float)
            maps = cv2.initUndistortRectifyMap(
                k,
                np.asarray(stream["distortion"], float),
                np.eye(3),
                k,
                tuple(stream["size"]),
                cv2.CV_32FC1,
            )
            info = dict(
                intrinsics=k.tolist(),
                view_from_target=transforms[name].tolist(),
                images={},
            )
            for index, rgb in enumerate(frames(stream, int(indices.max()) + 1)):
                if index not in indices:
                    continue
                frame_id = int(capture["frame_ids"][index])
                path = out / f"{name}_{frame_id:06d}.png"
                Image.fromarray(
                    cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR)
                ).save(path)
                info["images"][str(frame_id)] = dict(
                    path=path.name, sha256=digest(path)
                )
            views[name] = info
    write_json(
        out / "views.json",
        dict(
            schema="real2sim2real.hand-review-views/1",
            views=views,
            annotation_use="evaluation_only",
            frame_ids=list(map(int, requested)),
            calibration_sha256=digest(source.path("calibration.json")),
        ),
    )


def annotation_errors(annotations, predictions, frame_ids, views):
    """Score only explicitly visible, visually reviewed anchors; never fill labels."""
    from .camera import transform
    from .metric_hands import project

    rows = []
    for label in annotations.get("points", []):
        if label.get("visibility") != "visible":
            continue
        frame, hand, joint = (
            int(label["frame_id"]),
            int(label["hand"]),
            int(label["joint"]),
        )
        if hand not in (0, 1) or not 0 <= joint < 21 or label["view"] not in views:
            raise ValueError("Invalid annotated hand, joint or camera")
        index = np.flatnonzero(np.asarray(frame_ids) == frame)
        if len(index) != 1:
            raise ValueError("Annotation frame not in experiment")
        view = views[label["view"]]
        image = view["images"].get(str(frame))
        if image is None or label["image_sha256"] != image["sha256"]:
            raise ValueError("Annotation does not refer to the reviewed image")
        uv = np.asarray(label["xy_px"], float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("Invalid manual image coordinate")
        xyz = predictions[index[0], hand, joint]
        error = None
        if np.isfinite(xyz).all():
            point = transform(xyz[None], np.asarray(view["view_from_target"]))
            if point[0, 2] > 0:
                projected = project(point, np.asarray(view["intrinsics"]))[0]
                error = float(np.linalg.norm(projected - uv))
        rows.append(
            dict(
                frame_id=frame,
                hand=hand,
                joint=joint,
                view=label["view"],
                error_px=error,
                annotation_uncertainty_px=label.get("uncertainty_px"),
            )
        )
    return rows
