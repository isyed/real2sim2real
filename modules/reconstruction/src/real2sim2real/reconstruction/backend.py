"""RGB -> segmentation -> human/depth -> metric mesh -> observed object motion."""

from __future__ import annotations
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from real2sim2real.artifacts import digest, publish, write_json
from real2sim2real.stage import option_path


def engine_paths():
    import r2s2r_engines.cari4d.lib as library

    base = Path(library.__file__).parent
    engines = base.parents[1]
    return (
        base / "cari4d",
        engines / "sam3d_body/lib",
        engines / "foundation_pose/lib/FoundationPose",
    )


def cache_environment(phase, weights, sam3d_weights):
    """Keep object-model and human/depth model caches independent."""
    if phase == "metric_mesh":
        return dict(
            HF_HOME=str(sam3d_weights / "hf_home"),
            TORCH_HOME=str(sam3d_weights / "torch_home"),
        )
    return dict(
        HF_HOME=str(weights / "hf_home"),
        TORCH_HOME=str(weights / "sam3d_body/torch_home"),
    )


def run(request, phase, *, check_backend=False):
    import numpy as np
    from PIL import Image
    from .metric import pointmap, check_pointmap
    from .stage import GPU_PHASES

    if check_backend and phase not in GPU_PHASES:
        raise ValueError("Backend import checks apply to model phases only")

    output = Path(request["output"])
    video = Path(request["input"])
    options = request["options"]
    from .quality import QualityLimits, quality_report

    quality_limits = QualityLimits(**options.get("quality_limits", {}))
    quality_limits.validate()
    prompts = option_path(request, "prompts")
    from .mask_acceptance import enforce
    enforce(request, phase, check_backend=check_backend)
    if not check_backend and phase in ('metric_mesh', 'object_motion'):
        from .mask_features import enforce as enforce_features
        enforce_features(request)
    weights = option_path(request, "weights")
    sam2_weights = option_path(request, "sam2_weights")
    sam3d_weights = option_path(request, "sam3d_weights")
    human_ckpt = weights / "sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_model = (
        weights / "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    for path in [prompts, weights, sam2_weights, sam3d_weights, human_ckpt, mhr_model]:
        if not path.exists():
            raise FileNotFoundError(path)
    source, body, fp = engine_paths()
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, [source, body, fp]))
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        **cache_environment(phase, weights, sam3d_weights),
        MHR_ASSETS_ROOT=str(weights / "sam3d_body"),
        SAM3D_BODY_ROOT=str(body),
        FOUNDATIONPOSE_WEIGHTS_DIR=str(weights / "foundationpose/nvlabs_pytorch"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYOPENGL_PLATFORM="egl",
        OPENCV_IO_ENABLE_OPENEXR="1",
        R2S2R_FP_RENDER_BATCH="16",
        R2S2R_FP_REFINE_BATCH="32",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if phase == "object_motion":
        from .tracking_runtime import tracking_environment
        env.update(tracking_environment(weights / "foundationpose/nvlabs_pytorch"))
    os.environ.update(env)
    if check_backend:
        # Import the actual model CLI, not only this lightweight wrapper. A
        # missing transitive source package must fail before allocating a GPU.
        env["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in [fp, body, source]:
        sys.path.insert(0, str(path))
    local = output / "source.0.color.mp4"
    depth = output / "depth"
    export = output / "export/source"
    mesh_dir = output / "mesh"
    mask = output / "masks/1/000000.png"
    camera = "front_stereo_camera_left"

    def step(name, args):
        # The worker already runs inside the selected backend environment.
        subprocess.run(
            [sys.executable, *map(str, args), *(["--help"] if check_backend else [])],
            env=env,
            cwd=source,
            check=True,
        )

    if phase == "video":
        import av

        # Arbitrary user filenames are staged losslessly into the backend naming convention.
        local = output / "source.0.color.mp4"
        shutil.copy2(video, local)
        with av.open(str(local)) as capture:
            frames = [
                (frame.pts * frame.time_base, frame.to_image() if i == 0 else None)
                for i, frame in enumerate(capture.decode(video=0))
            ]
        times = np.asarray([float(x[0]) for x in frames])
        first = frames[0][1] if frames else None
        if first is None or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Video needs valid monotonic presentation timestamps")
        times -= times[0]
        count = len(times)
        write_json(
            output / "timeline.json", dict(frames=count, timestamps_s=times.tolist())
        )
        first.save(output / "first_rgb.png")
        from .prompt_review import review_first_frame

        review_first_frame(
            output / "first_rgb.png", prompts, output / "prompt_review",
            frame_count=count, source_video_sha256=request["input_sha256"],
        )
        return

    if phase == "segment":
        step(
            "segment",
            [
                "-m",
                "r2s2r_engines.sam2.lib.video_to_masks",
                "--video_path",
                local,
                "--prompts_path",
                prompts,
                "--masks_dir",
                output / "masks",
                "--weights_dir",
                sam2_weights,
            ],
        )
        return

    if phase == "pack_masks":
        import h5py
        from r2s2r_engines.cari4d.lib.pack_masks import masks_pack_cari4d_h5

        masks_pack_cari4d_h5(
            local, output / "masks/0", output / "masks/1", output / "masks.h5"
        )
        from .mask_review import review_masks

        review_masks(local,output/"masks",output/"mask_review",
            frame_count=json.loads((output/"timeline.json").read_text())["frames"],
            source_video_sha256=request["input_sha256"],prompts=prompts,
            frame_ids=options.get("mask_review_frames"))
        if options.get('mask_feature_checks') is not None:
            from .mask_features import check as check_features
            check_features(output,options['mask_feature_checks'],source_video_sha256=request['input_sha256'])
        return

    if phase == "depth":
        depth = output / "depth"
        if not check_backend:
            depth.mkdir()
        step(
            "depth",
            [
                source / "prep/mhr_wild_depth.py",
                "--video",
                local,
                "--output-depth",
                depth / "raw.npy",
                "--output-intrinsics",
                depth / "intrinsics.pkl",
                "--output-report",
                depth / "report.json",
                "--batch-size",
                "2",
                "--local-files-only",
            ],
        )
        return

    if phase == "export_human":
        from prep.prepare_mhr_wild_export import (
            prepare_mhr_wild_export,
            _write_object_template,
        )

        export = prepare_mhr_wild_export(
            local,
            output / "masks.h5",
            None,
            depth / "intrinsics.pkl",
            output / "export",
        )
        return

    if phase == "human":
        step(
            "human",
            [
                source / "prep/run_sam3d_mhr_export.py",
                export,
                "--out-file",
                output / "human.pkl",
                "--camera-id",
                "0",
                "--depth-root",
                export,
                "--sam3d-ckpt",
                human_ckpt,
                "--mhr-path",
                mhr_model,
                "--chunk-size",
                "4",
                "--sam3d-cache",
                output / "human_cache.h5",
                "--refit-batch-size",
                "64",
                "--human-prompt-mode",
                "sam2_bbox_mask",
                "--no-align-to-gt-depth",
            ],
        )
        return

    if phase == "depth_alignment":
        step(
            "depth_alignment",
            [
                source / "prep/align_depth_to_mhr_wild.py",
                export,
                "--raw-depth",
                depth / "raw.npy",
                "--depth-report",
                depth / "report.json",
                "--mhr-init",
                output / "human.pkl",
                "--output",
                depth / "aligned.h5",
                "--render-batch-size",
                "2",
                "--encoding-workers",
                "2",
                "--write-batch-size",
                "8",
            ],
        )
        return

    if phase == "metric_inputs":
        import h5py
        from .mesh_frame import extract_rgb, mask_path, select_mesh_frame

        camera = "front_stereo_camera_left"
        frame_count = json.loads((output / "timeline.json").read_text())["frames"]
        mesh_frame = select_mesh_frame(options.get("mesh_frame_index", 0), frame_count)
        metadata = json.loads((export / "wild_export.json").read_text())
        K = np.asarray(metadata["intrinsics"])
        with h5py.File(depth / "aligned.h5") as d:
            frames = d[f"aligned/{camera}"]
            if len(frames) != frame_count:
                raise ValueError("Aligned depth differs from the source timeline")
            z = (
                np.asarray(
                    Image.open(io.BytesIO(frames[mesh_frame].tobytes())),
                    dtype=np.float32,
                )
                / 1000
            )
        mask = mask_path(output, mesh_frame)
        if not mask.is_file():
            raise FileNotFoundError(mask)
        rgb = output / ("first_rgb.png" if mesh_frame == 0 else "mesh_input_rgb.png")
        if mesh_frame:
            size = extract_rgb(local, mesh_frame, rgb)
        else:
            with Image.open(rgb) as image:
                size = image.size
        with Image.open(mask) as mask_image:
            mask_size = mask_image.size
        if size != (z.shape[1], z.shape[0]) or mask_size != size:
            raise ValueError("Selected RGB, mask and depth have different native grids")
        xyz = pointmap(z, K)
        check = check_pointmap(xyz, K, np.asarray(Image.open(mask)) != 0)
        np.save(output / "pointmap.npy", xyz)
        camera_record = dict(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=z.shape[1],
            height=z.shape[0],
        )
        write_json(output / "camera.json", camera_record)
        write_json(output / "metric_input_check.json", dict(
            **check, mesh_frame_index=mesh_frame, source_video_sha256=request["input_sha256"],
            rgb_sha256=digest(rgb), mask_sha256=digest(mask),
            pointmap_sha256=digest(output / "pointmap.npy"),
            camera_sha256=digest(output / "camera.json"),
            aligned_depth_sha256=digest(depth / "aligned.h5")))
        return

    if phase == "metric_mesh":
        from .mesh_frame import mask_path, select_mesh_frame

        mesh_dir = output / "mesh"
        if not check_backend:
            mesh_dir.mkdir()
        if not check_backend:
            frame_count = json.loads((output / "timeline.json").read_text())["frames"]
            mesh_frame = select_mesh_frame(options.get("mesh_frame_index", 0), frame_count)
            check = json.loads((output / "metric_input_check.json").read_text())
            if check.get("mesh_frame_index") != mesh_frame:
                raise ValueError("Metric inputs were generated for a different mesh frame")
        else:
            mesh_frame = select_mesh_frame(options.get("mesh_frame_index", 0), 2**31)
        rgb = output / ("first_rgb.png" if mesh_frame == 0 else "mesh_input_rgb.png")
        mask = mask_path(output, mesh_frame)
        if not check_backend:
            for path, key in ((rgb, "rgb_sha256"), (mask, "mask_sha256"),
                              (output / "pointmap.npy", "pointmap_sha256"),
                              (output / "camera.json", "camera_sha256"),
                              (depth / "aligned.h5", "aligned_depth_sha256")):
                if digest(path) != check.get(key):
                    raise ValueError("Selected mesh input changed: " + str(path))
        # Separate process avoids keeping SAM3D and tracking models resident together.
        step(
            "metric_mesh",
            [
                "-m",
                "r2s2r_engines.sam3d.lib.image_to_mesh",
                "--image_path",
                rgb,
                "--mask_path",
                mask,
                "--mesh_path",
                mesh_dir / "canonical.glb",
                "--transform_path",
                mesh_dir / "transform.json",
                "--intrinsics_path",
                mesh_dir / "camera.json",
                "--weights_dir",
                sam3d_weights,
                "--seed",
                "0",
                "--pointmap_path",
                output / "pointmap.npy",
                "--pointmap_intrinsics_path",
                output / "camera.json",
            ],
        )
        return

    if phase == "object_export":
        from prep.prepare_mhr_wild_export import _write_object_template

        camera_record = json.loads((output / "camera.json").read_text())
        metadata = json.loads((export / "wild_export.json").read_text())
        from r2s2r_engines.sam3d.lib.autoscale_mesh import autoscale_mesh

        autoscale_mesh(
            mesh_dir / "canonical.glb",
            mesh_dir / "transform.json",
            mesh_dir / "scaled.glb",
        )
        if json.loads((mesh_dir / "camera.json").read_text()) != camera_record:
            raise ValueError("Mesh generator changed the shared camera")
        transform, extents = _write_object_template(
            mesh_dir / "scaled.glb", export / "object_mesh/output_aligned.glb"
        )
        metadata.pop("object_geometry_pending", None)
        metadata.update(
            object_mesh_file=str(export / "object_mesh/output_aligned.glb"),
            source_object_mesh_to_aligned_transform=transform.tolist(),
            aligned_object_extents_m=extents.tolist(),
        )
        from lib_mhr import stamp_object_pose_frame_metadata

        write_json(
            export / "wild_export.json", stamp_object_pose_frame_metadata(metadata)
        )
        return

    if phase == "object_motion":
        step(
            "object_motion",
            [
                source / "prep/run_foundationpose_mhr_export.py",
                export,
                "--depth-root",
                depth / "aligned.h5",
                "--out-file",
                output / "object_motion.pkl",
                "--camera-id",
                "0",
                "--iteration",
                "5",
                "--max-attempts",
                "3",
                "--register-first-then-track",
                "--no-first-usable-frame-gt-rotation-oracle",
            ],
        )
        return

    if phase == "publish":
        import trimesh

        timeline = json.loads((output / "timeline.json").read_text())
        count = timeline["frames"]
        times = np.asarray(timeline["timestamps_s"])
        K = np.asarray(
            json.loads((export / "wild_export.json").read_text())["intrinsics"]
        )
        check = json.loads((output / "metric_input_check.json").read_text())
        with (output / "object_motion.pkl").open("rb") as stream:
            motion = pickle.load(stream)
        with (output / "human.pkl").open("rb") as stream:
            human = pickle.load(stream)
        from real2sim2real.human import (
            normalize_predicted_object_rotations,
            decode_initialization_mhr,
            load_mhr_buffers,
        )

        poses, roundoff = normalize_predicted_object_rotations(
            np.asarray(motion["fp_poses_cam"])[:, 0]
        )
        if len(poses) != count:
            raise ValueError("Object motion does not cover the video timeline")
        packed = decode_initialization_mhr(human, load_mhr_buffers(human_ckpt))
        if len(packed["mhr_model_params"]) != count:
            raise ValueError("Human motion does not cover the video timeline")
        np.savez_compressed(output / "human.npz", **packed)
        mesh = trimesh.load(
            export / "object_mesh/output_aligned.glb", force="mesh", process=False
        )
        geometry = dict(
            vertices=mesh.vertices,
            faces=mesh.faces,
            object_rotation=poses[:, :3, :3],
            object_translation=poses[:, :3, 3],
            object_scale=np.array(1.0),
            frame_ids=np.arange(count),
            timestamps_s=times,
            intrinsics=np.broadcast_to(K, (count, 3, 3)),
            world_to_camera=np.broadcast_to(np.eye(4), (count, 4, 4)),
        )
        np.savez_compressed(output / "geometry.npz", **geometry)
        quality = quality_report(
            poses,
            times,
            image_iou=motion.get("fp_full_image_iou"),
            tracker_reliable=motion.get("fp_reliable"),
            limits=quality_limits,
        )
        quality["sources"] = dict(
            geometry_sha256=digest(output / "geometry.npz"),
            tracking_sha256=digest(output / "object_motion.pkl"),
        )
        write_json(output / "quality_report.json", quality)
        write_json(
            output / "reconstruction.json",
            dict(
                pipeline="shared-metric-depth/1",
                frames=count,
                coordinates="opencv_camera_world_metres",
                metric_pointmap=check,
                mesh_and_tracking_depth_sha256=digest(depth / "aligned.h5"),
                roundoff=roundoff,
                tracking_reliable_frames=int(
                    np.asarray(motion.get("fp_reliable", np.zeros(count, bool))).sum()
                ),
                human_model_sha256=digest(human_ckpt),
                mhr_model_sha256=digest(mhr_model),
                physical_execution_validated=False,
                metric_accuracy_is_prediction=True,
            ),
        )
        files = {
            "geometry": output / "geometry.npz",
            "human": output / "human.npz",
            "timeline": output / "timeline.json",
            "mesh": export / "object_mesh/output_aligned.glb",
            "diagnostics": output / "reconstruction.json",
            "quality": output / "quality_report.json",
            "depth": depth / "aligned.h5",
            "object_masks": export / f"object_masks/{camera}.h5",
            "human_masks": export / f"human_masks/{camera}.h5",
        }
        publish(
            output,
            "reconstruction",
            files,
            metadata=dict(
                sequence_id=options.get("sequence_id", video.stem),
                world_frame_id="camera",
                source_video_sha256=request["input_sha256"],
                physical_execution_validated=False,
            ),
        )
        return
    raise ValueError("Unknown reconstruction phase: " + phase)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help="Check the full model CLI imports with CUDA hidden, without inference",
    )
    args = parser.parse_args()
    from .mask_acceptance import MaskReviewRequired
    try:
        run(json.loads(args.request.read_text()),args.phase,check_backend=args.check_backend)
    except MaskReviewRequired as error:
        print('MASK REVIEW REQUIRED: '+str(error),file=sys.stderr)
        raise SystemExit(78)
