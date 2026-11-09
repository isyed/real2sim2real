"""Optional RSS scene descriptions and actual RSS camera parity checks.

Scene construction here creates specifications only, not a physics solver.
No robot, grasp, support, mass, inertia or friction is inferred from a video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import load
from .bundle import exchange_poses, read_geometry, rigid


def camera_parameters(intrinsics, world_to_camera, source_to_rss, *, width, height):
    camera_to_rss = rigid(source_to_rss) @ np.linalg.inv(rigid(world_to_camera))
    quaternion_xyzw = Rotation.from_matrix(camera_to_rss[:3, :3]).as_quat()
    k = np.asarray(intrinsics)
    return dict(name="source_rgb", width=int(width), height=int(height),
                fx=float(k[0, 0]), fy=float(k[1, 1]),
                cx=float(k[0, 2]), cy=float(k[1, 2]),
                xform=tuple(camera_to_rss[:3, 3]) + tuple(quaternion_xyzw))


def inspection_scene(bundle, frame_id):
    """Build RSS specs for an object-only, kinematic source inspection view.

    Never call this a dynamic task rollout. Support and human alignment must
    be separately established before a later physical-scene adapter is used.
    """
    import robosimstudio as rss
    bundle = Path(bundle)
    receipt = load(bundle, kinds={"rss-source-exchange"})
    metadata = receipt["metadata"]
    geometry = read_geometry(bundle / receipt["files"]["geometry"]["path"])
    indices = np.flatnonzero(geometry["frame_ids"] == frame_id)
    if len(indices) != 1:
        raise ValueError("Unknown source frame ID")
    index = int(indices[0])
    transform = np.asarray(metadata["source_to_rss"])
    pose = transform @ exchange_poses(geometry)[index]
    scene = rss.Scene(cell="void", name="Source inspection; not a task rollout")
    scene.add(rss.body(name="source_object", asset=rss.AssetSpec(
        path=str((bundle / receipt["files"]["exchange_mesh"]["path"]).resolve()),
        scale=float(geometry["object_scale"])),
        at=tuple(pose[:3, 3]), quat=tuple(Rotation.from_matrix(pose[:3, :3]).as_quat()),
        static=True))
    camera = rss.CameraSpec(**camera_parameters(
        geometry["intrinsics"][index], geometry["world_to_camera"][index],
        transform, width=metadata["width"], height=metadata["height"]))
    scene.add(camera)
    return scene, camera


def camera_parity(bundle, frame_ids):
    """Check source pixels against the installed RSS camera implementation."""
    bundle = Path(bundle)
    receipt = load(bundle, kinds={"rss-source-exchange"})
    geometry = read_geometry(bundle / receipt["files"]["geometry"]["path"])
    transform = np.asarray(receipt["metadata"]["source_to_rss"])
    rows = []
    for frame_id in frame_ids:
        indices = np.flatnonzero(geometry["frame_ids"] == frame_id)
        if len(indices) != 1:
            raise ValueError("Unknown source frame ID")
        index = int(indices[0])
        _, camera = inspection_scene(bundle, frame_id)
        vertices = geometry["vertices"][::max(1, len(geometry["vertices"]) // 500)]
        world = (vertices * geometry["object_scale"]) @ geometry["object_rotation"][index].T
        world += geometry["object_translation"][index]
        extrinsic = geometry["world_to_camera"][index]
        cv = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        front = cv[:, 2] > 1e-6
        if not front.any():
            raise ValueError("No positive-depth mesh samples to validate")
        pixels = cv @ geometry["intrinsics"][index].T
        pixels = pixels[front, :2] / pixels[front, 2, None]
        rss_points = world @ transform[:3, :3].T + transform[:3, 3]
        projected, depth = camera.project(rss_points[front])
        pixel_error = float(np.max(np.abs(projected - pixels)))
        depth_error = float(np.max(np.abs(depth - cv[front, 2])))
        rows.append(dict(frame_id=int(frame_id), max_pixel_error=pixel_error,
                         max_depth_error_m=depth_error,
                         passed=pixel_error < 1e-6 and depth_error < 1e-8))
    if not rows:
        raise ValueError("At least one test frame is required")
    return dict(passed=all(row["passed"] for row in rows), frames=rows,
                test="actual_rss_camera_projection_cpu", physics_tested=False)
