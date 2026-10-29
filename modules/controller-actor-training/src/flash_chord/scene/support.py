# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add a sequence's support surfaces (kinematic collision shapes) from its support USDA.

The support USDA holds primitive supports or metre-scale reconstructed triangle
meshes. They are added as static (kinematic) collision shapes. Newton cylinders
extend along Z, matching the USDA's ``axis = "Z"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import newton
import numpy as np

from flash_chord.scene.collision import ShapeSpan


@dataclass(frozen=True)
class SupportBinding:
    """Resolved body, free-root coordinates, and shapes owned by one support component."""

    body_id: int
    free_q_ids: tuple[int, ...]
    free_dof_ids: tuple[int, ...]
    shapes: ShapeSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_q_ids", tuple(self.free_q_ids))
        object.__setattr__(self, "free_dof_ids", tuple(self.free_dof_ids))
        if self.body_id < 0:
            raise ValueError("support body ID must be nonnegative")
        if (
            len(self.free_q_ids) != 7
            or self.free_q_ids != tuple(range(self.free_q_ids[0], self.free_q_ids[0] + 7))
            or self.free_q_ids[0] < 0
        ):
            raise ValueError("support must bind exactly seven contiguous nonnegative free-joint q IDs")
        if (
            len(self.free_dof_ids) != 6
            or self.free_dof_ids != tuple(range(self.free_dof_ids[0], self.free_dof_ids[0] + 6))
            or self.free_dof_ids[0] < 0
        ):
            raise ValueError("support must bind exactly six contiguous nonnegative free-joint DOF IDs")
        if not len(self.shapes):
            raise ValueError("support must own at least one shape")


def _mesh_geometry(prim, stage):
    """Validate the complete authored surface before changing the builder."""
    from pxr import Gf, UsdGeom

    if not stage.HasAuthoredMetadata("metersPerUnit") or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("reconstructed support mesh must explicitly use metre units")
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("reconstructed support mesh must use Z-up world coordinates")
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or not np.isfinite(vertices).all() or counts.ndim != 1 or not len(counts)
            or np.any(counts < 3) or indices.ndim != 1 or counts.sum() != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(vertices))):
        raise ValueError(f"invalid support mesh topology or vertices: {prim.GetPath()}")
    if mesh.GetHoleIndicesAttr().Get():
        raise ValueError("support mesh holes require explicit collision triangulation")
    if np.any(counts != 3):
        raise ValueError("support mesh must contain explicit triangles")
    # Includes transforms on ancestor Xforms, not only the Mesh's own translate.
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.asarray([transform.Transform(Gf.Vec3d(*v)) for v in vertices], dtype=np.float64)
    faces = indices.reshape(-1, 3)
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        faces = faces[:, ::-1]
    if np.linalg.det(np.asarray(transform)[:3, :3]) < 0:
        faces = faces[:, ::-1]
    edges = vertices[faces[:, 1:]] - vertices[faces[:, :1]]
    if (not np.isfinite(vertices).all()
            or np.any(np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1) <= 1e-12)):
        raise ValueError("support mesh contains degenerate or non-finite triangles")
    return vertices.astype(np.float32), faces.reshape(-1).astype(np.int32)


def add_support_surfaces(builder: newton.ModelBuilder, support_usda_path) -> SupportBinding | None:
    """Add the support USDA's prims as kinematic collision shapes on one static body.

    Returns the exact component topology. An explicitly requested missing/invalid file fails.
    """
    path = Path(support_usda_path)
    if not path.is_file():
        raise FileNotFoundError(f"support surface does not exist: {path}")

    import warp as wp
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"failed to open support surface: {path}")
    shapes = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in ("Cylinder", "Cube", "Mesh"):
            continue
        if prim.GetAttribute("physics:collisionEnabled").Get() is False:
            continue
        if type_name == "Mesh":
            shapes.append((type_name, prim.GetName(), (0., 0., 0.), _mesh_geometry(prim, stage)))
            continue
        translate = prim.GetAttribute("xformOp:translate").Get()
        position = (
            (0.0, 0.0, 0.0) if translate is None else (float(translate[0]), float(translate[1]), float(translate[2]))
        )
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"support shape {prim.GetPath()} has a non-finite translation")
        if type_name == "Cylinder":
            dimensions = (
                float(prim.GetAttribute("radius").Get()),
                float(prim.GetAttribute("height").Get()),
            )
        else:
            size = prim.GetAttribute("size").Get()
            scale = prim.GetAttribute("xformOp:scale").Get()
            scale = (1.0, 1.0, 1.0) if scale is None else tuple(float(value) for value in scale)
            if len(scale) != 3 or not all(math.isfinite(value) and value != 0.0 for value in scale):
                raise ValueError(f"support shape {prim.GetPath()} must have a finite nonzero scale")
            edge = float(size if size is not None else 2.0)
            dimensions = tuple(edge * abs(value) for value in scale)
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError(f"support shape {prim.GetPath()} must have positive finite dimensions")
        shapes.append((type_name, prim.GetName(), position, dimensions))
    if not shapes:
        # Released ground-only episodes contain an empty Xform stage. The
        # scene's ground plane already provides their only support collider.
        if all(prim.GetTypeName() in ('Xform', 'Scope', '') for prim in stage.Traverse()):
            return None
        raise ValueError(f"support surface contains no Cylinder, Cube or Mesh collision shapes: {path}")

    body_start = builder.body_count
    joint_start = builder.joint_count
    q_start = builder.joint_coord_count
    dof_start = builder.joint_dof_count
    shape_start = builder.shape_count
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
        label="support",
    )
    for type_name, name, position, dimensions in shapes:
        xform = wp.transform(wp.vec3(*position), wp.quat_identity())
        if type_name == "Cylinder":
            radius, height = dimensions
            builder.add_shape_cylinder(
                body,
                xform=xform,
                radius=radius,
                half_height=0.5 * height,
                label=name,
            )
        elif type_name == "Mesh":
            vertices, indices = dimensions
            mesh = newton.Mesh(vertices, indices, compute_inertia=False)
            builder.add_shape_mesh(body=body, xform=xform, mesh=mesh, label=name,
                                   cfg=replace(builder.default_shape_cfg, density=0.0))
        else:  # Cube: USD `size * xformOp:scale` gives the full dimensions
            hx, hy, hz = (0.5 * value for value in dimensions)
            builder.add_shape_box(body, xform=xform, hx=hx, hy=hy, hz=hz, label=name)
    if (
        body != body_start
        or builder.body_count != body_start + 1
        or builder.joint_count != joint_start + 1
        or builder.joint_coord_count != q_start + 7
        or builder.joint_dof_count != dof_start + 6
    ):
        raise RuntimeError("support body creation did not append one Newton free-root component")
    joint = joint_start
    if (
        builder.joint_type[joint] != newton.JointType.FREE
        or builder.joint_parent[joint] != -1
        or builder.joint_child[joint] != body
        or builder.joint_q_start[joint] != q_start
        or builder.joint_qd_start[joint] != dof_start
    ):
        raise RuntimeError("support body does not expose the expected world-parented Newton free joint")
    return SupportBinding(
        body_id=body,
        free_q_ids=tuple(range(q_start, q_start + 7)),
        free_dof_ids=tuple(range(dof_start, dof_start + 6)),
        shapes=ShapeSpan(shape_start, builder.shape_count),
    )
