# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

from r2s2r_engines.common.datatypes import CameraIntrinsics, Mask
from r2s2r_engines.mesh.lib.mesh import Mesh
from r2s2r_engines.mesh.lib.mesh_render_depth import mesh_render_depth
import numpy as np


def mesh_render_mask(mesh: Mesh, intrinsics: CameraIntrinsics) -> Mask:
    """
    Render a binary silhouette mask of the mesh from a camera at the origin.

    The camera convention matches mesh_render_depth (OpenCV: looking along +Z, Y down).
    Mask pixels are 1.0 where the mesh is visible, 0.0 for background.
    """
    depth_img = mesh_render_depth(mesh, intrinsics)
    return Mask(mask=(depth_img.depth > 0))
