"""Isolated model workers. No reference motion or reference scan is accepted."""

from pathlib import Path
import argparse
import json
import os
import numpy as np
from PIL import Image
from real2sim2real.artifacts import write_json


def segmentation(request, check=False):
    from r2s2r_engines.sam2.lib.video_to_masks import video_to_masks

    if check:
        return
    out = Path(request["output"])
    # The SAM2 reader accepts numbered frame directories without video re-encoding.
    video_to_masks(
        str(out / "rgb"),
        str(out / "prompts.json"),
        str(out / "masks"),
        request["weights"]["sam2"],
    )


def mesh(request, check=False):
    from r2s2r_engines.sam3d.lib.image_to_mesh import image_to_mesh
    from r2s2r_engines.sam3d.lib.autoscale_mesh import autoscale_mesh
    from .camera import unproject

    if check:
        return
    out = Path(request["output"])
    folder = out / "mesh"
    folder.mkdir(exist_ok=True)
    index = int(request.get("mesh_frame", 0))
    timeline = json.loads((out / "timeline.json").read_text())
    depth = np.load(out / "depth" / f"{index:06d}.npy")
    object_id = request.get("object_id", 1)
    mask_path = out / "masks" / str(object_id) / f"{index:06d}.png"
    mask = np.asarray(Image.open(mask_path)) > 0
    coverage = float(((depth > 0) & mask).sum() / max(mask.sum(), 1))
    if coverage < request.get("minimum_object_depth_coverage", 0.35):
        raise ValueError(f"Insufficient measured object depth: {coverage:.3f}")
    points = unproject(depth, np.asarray(timeline["intrinsics"]))
    points[depth <= 0] = np.nan
    np.save(folder / "pointmap.npy", points.astype(np.float32))
    write_json(
        folder / "input_quality.json",
        dict(
            frame=index,
            object_depth_coverage=coverage,
            depth_source="calibrated_stereo",
            ground_truth_used=False,
        ),
    )
    image_to_mesh(
        str(out / "rgb" / f"{index:06d}.png"),
        str(mask_path),
        str(folder / "canonical.glb"),
        str(folder / "transform.json"),
        str(folder / "camera.json"),
        weights_dir=request["weights"]["sam3d"],
        seed=0,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        pointmap_path=str(folder / "pointmap.npy"),
        pointmap_intrinsics_path=str(out / "camera.json"),
    )
    autoscale_mesh(
        folder / "canonical.glb", folder / "transform.json", folder / "metric.glb"
    )


def tracking(request, check=False):
    from real2sim2real.reconstruction.tracking_runtime import tracking_environment

    os.environ.update(tracking_environment(request["weights"]["foundationpose"]))
    from r2s2r_engines.foundation_pose.lib.foundation_pose_tracker import (
        FoundationPoseTracker,
    )
    from r2s2r_engines.mesh.lib.mesh import Mesh
    from r2s2r_engines.common.datatypes import (
        CameraIntrinsics,
        DepthImage,
        Mask,
        Image as RGBImage,
        Transform3d,
    )
    from Utils import nvdiffrast_render, mycpp

    if mycpp is None or not callable(getattr(mycpp, "cluster_poses", None)):
        raise ImportError(
            "FoundationPose native extension is unavailable; configure native_python_paths for this image"
        )
    import torch
    import cv2
    from scipy.spatial.transform import Rotation
    from .tracking import refine_with_retries, refinement_schedule

    iterations = request.get("tracking_iterations", 5)
    retry_iterations = request.get("retry_iterations", [])
    refinement_schedule(iterations, retry_iterations)

    if check:
        return
    out = Path(request["output"])
    timeline = json.loads((out / "timeline.json").read_text())
    count = timeline["frames"]
    world_from_camera = np.load(out / "camera_motion.npz")["world_from_camera"]
    camera = CameraIntrinsics.load(str(out / "camera.json"))
    k = camera.to_matrix()
    tracker = FoundationPoseTracker(
        Mesh.load(str(out / "mesh/metric.glb")),
        request["weights"]["foundationpose"],
        backend="nvlabs_pytorch",
    )
    poses = np.full((count, 4, 4), np.nan)
    raw = poses.copy()
    accepted = np.zeros(count, bool)
    records = []
    (out / "overlays").mkdir(exist_ok=True)
    limits = dict(
        min_iou=0.5,
        max_depth_error_m=0.08,
        min_depth_pixels=100,
        max_speed_m_s=2.0,
        max_angular_speed_deg_s=600.0,
    )
    limits.update(request.get("tracking_limits", {}))
    for i in range(count):
        rgb = np.asarray(Image.open(out / "rgb" / f"{i:06d}.png").convert("RGB"))
        depth = np.load(out / "depth" / f"{i:06d}.npy").astype(np.float32)
        mask = (
            np.asarray(
                Image.open(
                    out / "masks" / str(request.get("object_id", 1)) / f"{i:06d}.png"
                )
            )
            > 0
        )
        ignored = np.zeros_like(mask)
        for hand_id in request.get("occluder_ids", []):
            path = out / "masks" / str(hand_id) / f"{i:06d}.png"
            if path.exists():
                ignored |= np.asarray(Image.open(path)) > 0
        visible = ~ignored
        rendered = None

        def assess(pose):
            nonlocal rendered
            est = tracker._est
            centred = torch.as_tensor(
                pose, device="cuda", dtype=torch.float32
            ) @ torch.linalg.inv(est.get_tf_to_centered_mesh())
            _, rendered_depth, _ = nvdiffrast_render(
                K=k,
                H=rgb.shape[0],
                W=rgb.shape[1],
                ob_in_cams=centred[None],
                glctx=tracker._glctx,
                mesh_tensors=est.mesh_tensors,
            )
            rd = rendered_depth[0].detach().cpu().numpy()
            rendered = rd > 0
            intersection = rendered & mask & visible
            union = (rendered | mask) & visible
            iou = float(intersection.sum() / max(union.sum(), 1))
            measured = intersection & (depth > 0)
            depth_error = (
                float(np.median(np.abs(rd[measured] - depth[measured])))
                if measured.any()
                else None
            )
            rejected = []
            if iou < limits["min_iou"]:
                rejected.append("silhouette")
            if (
                measured.sum() < limits["min_depth_pixels"]
                or depth_error is None
                or depth_error > limits["max_depth_error_m"]
            ):
                rejected.append("depth")
            speed = angular_speed = None
            if i:
                dt = timeline["timestamps_s"][i] - timeline["timestamps_s"][i - 1]
                a = world_from_camera[i - 1] @ poses[i - 1]
                b = world_from_camera[i] @ pose
                speed = float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / dt)
                angular_speed = float(
                    np.rad2deg(
                        Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
                    )
                    / dt
                )
                if (
                    speed > limits["max_speed_m_s"]
                    or angular_speed > limits["max_angular_speed_deg_s"]
                ):
                    rejected.append("motion")
            return dict(
                frame=i,
                accepted=not rejected,
                iou=iou,
                depth_error_m=depth_error,
                depth_pixels=int(measured.sum()),
                speed_m_s=speed,
                angular_speed_deg_s=angular_speed,
                rejected_for=rejected,
            )

        if i == 0:
            pose = tracker.register(
                RGBImage(rgb), DepthImage(depth), Mask(mask), camera, iteration=5
            ).to_matrix()
            record = assess(pose)
            attempts = [dict(iterations=5, pose=pose.tolist(), quality=record.copy())]
        else:
            pose, record, attempts = refine_with_retries(
                poses[i - 1],
                lambda steps: tracker.track_one(
                    RGBImage(rgb), DepthImage(depth), camera, iteration=steps
                ).to_matrix(),
                assess,
                lambda seed: tracker.reset_to_pose(Transform3d.from_matrix(seed)),
                primary=iterations,
                retries=retry_iterations,
            )
        raw[i] = np.asarray(attempts[0]["pose"])
        passed = record["accepted"]
        iou = record["iou"]
        rejected = record["rejected_for"]
        if passed:
            poses[i] = pose
            accepted[i] = True
        record["attempts"] = attempts
        record["recovered"] = passed and len(attempts) > 1
        records.append(record)
        preview = rgb.copy()
        preview[rendered] = (
            preview[rendered] * 0.5 + np.array([0, 220, 100]) * 0.5
        ).astype(np.uint8)
        cv2.putText(
            preview,
            f"{i}: IoU {iou:.3f} {'PASS' if passed else 'REJECT'}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 50, 50),
            2,
        )
        Image.fromarray(preview).save(out / "overlays" / f"{i:06d}.jpg")
        np.savez_compressed(
            out / "object_motion.npz",
            camera_from_object=poses,
            raw_camera_from_object=raw,
            accepted=accepted,
            frame_ids=timeline["frame_ids"],
            timestamps_s=timeline["timestamps_s"],
        )
        write_json(
            out / "tracking_quality.json",
            dict(
                accepted_frames=int(accepted.sum()),
                total_frames=count,
                records=records,
                limits=limits,
                refinement_schedule=[iterations, *retry_iterations],
                ground_truth_used=False,
                complete=bool(accepted.all()),
            ),
        )
        print(json.dumps(record), flush=True)
        if not passed:
            # Preserve raw diagnostics; do not propagate the rejected state.
            raise ValueError(f"Object tracking rejected frame {i}: {rejected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--phase",
        choices=["segment", "mesh", "track", "hands", "refine-hands", "temporal-hands"],
        required=True,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    from .hamer_worker import run as hands
    from .hand_refinement import run as refine_hands
    from .temporal_hand_worker import run as temporal_hands

    {
        "segment": segmentation,
        "mesh": mesh,
        "track": tracking,
        "hands": hands,
        "refine-hands": refine_hands,
        "temporal-hands": temporal_hands,
    }[args.phase](request, args.check)


if __name__ == "__main__":
    main()
