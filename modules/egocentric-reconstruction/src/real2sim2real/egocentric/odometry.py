"""Metric background RGB-D odometry; a failed pose never becomes a new seed."""

from pathlib import Path
import json
import cv2
import numpy as np
from real2sim2real.artifacts import write_json
from .camera import unproject


def estimate(
    output,
    *,
    feature_roi=(0, 0, 1, 1),
    exclusion_ids=(1,),
    max_step_m=0.15,
    max_step_deg=15,
):
    output = Path(output)
    timeline = json.loads((output / "timeline.json").read_text())
    k = np.asarray(timeline["intrinsics"], float)
    count = timeline["frames"]
    poses = np.full((count, 4, 4), np.nan)
    poses[0] = np.eye(4)
    accepted = np.zeros(count, bool)
    accepted[0] = True
    records = [dict(frame=0, accepted=True, origin="first_target_camera")]
    previous = cv2.imread(str(output / "rgb/000000.png"), cv2.IMREAD_GRAYSCALE)
    height, width = previous.shape
    roi = np.array(feature_roi, float)
    if (
        roi.shape != (4,)
        or not np.isfinite(roi).all()
        or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)
    ):
        raise ValueError("Invalid normalized static-feature ROI")
    for i in range(1, count):
        current = cv2.imread(str(output / "rgb" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        previous_depth = np.load(output / "depth" / f"{i - 1:06d}.npy")
        mask = np.zeros_like(previous)
        x0, y0, x1, y1 = (roi * [width, height, width, height]).astype(int)
        mask[y0:y1, x0:x1] = 255
        mask[previous_depth <= 0] = 0
        for object_id in exclusion_ids:
            path = output / "masks" / str(object_id) / f"{i - 1:06d}.png"
            if path.exists():
                excluded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                mask[cv2.dilate(excluded, np.ones((21, 21), np.uint8)) > 0] = 0
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=1800, qualityLevel=0.01, minDistance=8, mask=mask
        )
        if points is None or len(points) < 30:
            records.append(
                dict(frame=i, accepted=False, reason="insufficient_background_features")
            )
            break
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, winSize=(21, 21), maxLevel=3
        )
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, tracked, None, winSize=(21, 21), maxLevel=3
        )
        valid = (
            (status.ravel() > 0)
            & (back_status.ravel() > 0)
            & (np.linalg.norm(points[:, 0] - back[:, 0], axis=1) < 1.0)
        )
        uv = np.rint(points[:, 0]).astype(int)
        xyz = unproject(previous_depth, k)[uv[:, 1], uv[:, 0]]
        valid &= (xyz[:, 2] > 0.15) & (xyz[:, 2] < 8)
        if valid.sum() < 30:
            records.append(
                dict(
                    frame=i,
                    accepted=False,
                    reason="insufficient_metric_correspondences",
                )
            )
            break
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            xyz[valid],
            tracked[valid, 0],
            k,
            None,
            iterationsCount=200,
            reprojectionError=2,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 25:
            records.append(dict(frame=i, accepted=False, reason="pnp_failed"))
            break
        indices = inliers.ravel()
        rv, tv = cv2.solvePnPRefineLM(
            xyz[valid][indices], tracked[valid, 0][indices], k, None, rv, tv
        )
        step = np.eye(4)
        step[:3, :3] = cv2.Rodrigues(rv)[0]
        step[:3, 3] = tv.ravel()
        error = np.linalg.norm(
            cv2.projectPoints(xyz[valid][indices], rv, tv, k, None)[0][:, 0]
            - tracked[valid, 0][indices],
            axis=1,
        )
        translation = float(np.linalg.norm(tv))
        angle = float(np.rad2deg(np.linalg.norm(rv)))
        passed = (
            translation <= max_step_m
            and angle <= max_step_deg
            and np.median(error) < 1.5
        )
        records.append(
            dict(
                frame=i,
                accepted=bool(passed),
                inliers=len(indices),
                translation_m=translation,
                rotation_deg=angle,
                median_reprojection_px=float(np.median(error)),
            )
        )
        if not passed:
            break
        poses[i] = poses[i - 1] @ np.linalg.inv(step)
        accepted[i] = True
        previous = current
        print(f"camera frame {i}/{count - 1}: {len(indices)} inliers", flush=True)
    np.savez_compressed(
        output / "camera_motion.npz",
        world_from_camera=poses,
        accepted=accepted,
        frame_ids=timeline["frame_ids"],
        timestamps_s=timeline["timestamps_s"],
    )
    write_json(
        output / "camera_motion_quality.json",
        dict(
            method="background_metric_rgbd_pnp",
            ground_truth_used=False,
            accepted_frames=int(accepted.sum()),
            total_frames=count,
            records=records,
        ),
    )
    if not accepted.all():
        raise ValueError(
            "Camera motion incomplete; inspect quality report before further inference"
        )
    return poses
