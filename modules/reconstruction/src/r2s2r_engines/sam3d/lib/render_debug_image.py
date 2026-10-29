# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import cv2
import trimesh
import json
import numpy as np


def wxyz_quat_to_rotation_matrix(w, x, y, z):
    rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return rotation_matrix

def render_debug_image(
        image_path: str,
        mesh_path: str,
        transform_path: str,
        intrinsics_path: str,
        output_image_path: str,
        num_vertices_to_use: int = 5000
    ):

    with open(transform_path, "r") as f:
        transform = json.load(f)
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    scene = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene

    camera_intrinsics = intrinsics

    object_translation = np.array(transform['translation'])
    object_rotation = np.array(transform['rotation'])
    object_scale = np.array(transform['scale'])

    rotation_matrix = wxyz_quat_to_rotation_matrix(*object_rotation)
    vertices_3d = np.asarray(mesh.vertices) * object_scale[None, :]
    vertices_3d = vertices_3d @ rotation_matrix.T + object_translation[None, :]

    z = vertices_3d[:, 2]
    valid = np.isfinite(vertices_3d).all(axis=1) & (z > 0)
    x = vertices_3d[valid, 0] / z[valid]
    y = vertices_3d[valid, 1] / z[valid]

    x_image = x * camera_intrinsics['fx'] + camera_intrinsics['cx']
    y_image = y * camera_intrinsics['fy'] + camera_intrinsics['cy']

    vertices_2d = np.stack([x_image, y_image], axis=1)

    image = cv2.imread(image_path)
    visible = (
        np.isfinite(vertices_2d).all(axis=1)
        & (vertices_2d[:, 0] >= 0)
        & (vertices_2d[:, 0] < image.shape[1])
        & (vertices_2d[:, 1] >= 0)
        & (vertices_2d[:, 1] < image.shape[0])
    )
    visible_vertices = vertices_2d[visible]
    for vertex in visible_vertices[np.random.permutation(len(visible_vertices))[:num_vertices_to_use]]:
        cv2.circle(image, (int(vertex[0]), int(vertex[1])), 1, (255, 0, 0), -1)

    cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a debug image of a mesh")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the mesh")
    parser.add_argument("--transform_path", type=str, required=True, help="Path to the transform")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Path to the intrinsics")
    parser.add_argument("--output_image_path", type=str, required=True, help="Path to the output image")
    parser.add_argument("--num_vertices_to_use", type=int, default=5000, help="Number of vertices to use")
    args = parser.parse_args()
    render_debug_image(args.image_path, args.mesh_path, args.transform_path, args.intrinsics_path, args.output_image_path, args.num_vertices_to_use)
