"""Finite observed surface connectivity without filling arbitrary scene holes."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

from .camera import intrinsic, rigid


def bearing_skin(
    points_world_m,
    anchor_world_from_camera,
    k,
    visible_mask,
    *,
    pixel_cell=2,
    maximum_pixel_edge=12.0,
    maximum_metric_edge_m=0.025,
):
    """Fuse samples in anchor-image cells and triangulate bounded local bridges.

    This produces a zero-thickness skin, not a closed collider. Medians reduce
    repeated-view noise; cell depth ranges remain explicit. Masked-out gaps,
    long image bridges and metric depth jumps cannot become bearing triangles.
    """
    points = np.asarray(points_world_m, float)
    T = rigid(anchor_world_from_camera)
    K = intrinsic(k)
    mask = np.asarray(visible_mask, bool)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
        or mask.ndim != 2
        or not isinstance(pixel_cell, int)
        or pixel_cell < 1
        or not np.isfinite([maximum_pixel_edge, maximum_metric_edge_m]).all()
        or maximum_pixel_edge <= 0
        or maximum_metric_edge_m <= 0
    ):
        raise ValueError(
            "Finite points, image mask and positive bridge bounds required"
        )
    camera = (points - T[:3, 3]) @ T[:3, :3]
    q = camera @ K.T
    uv = q[:, :2] / np.maximum(q[:, 2:], 1e-12)
    rounded = np.rint(uv).astype(int)
    h, w = mask.shape
    inside = (
        (camera[:, 2] > 0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < w)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < h)
    )
    indices = np.flatnonzero(inside)
    indices = indices[mask[rounded[indices, 1], rounded[indices, 0]]]
    camera = camera[indices]
    uv = uv[indices]
    cells = np.floor(uv / pixel_cell).astype(int)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    if len(keys) < 3:
        raise ValueError("Too few visible cells")
    vertices = []
    pixels = []
    ranges = []
    counts = []
    for i in range(len(keys)):
        p = camera[inverse == i]
        v = np.median(p, axis=0)
        vertices.append(v)
        q = K @ v
        pixels.append(q[:2] / q[2])
        ranges.append(np.ptp(p[:, 2]))
        counts.append(len(p))
    pixels = np.array(pixels)
    vertices = np.array(vertices)
    faces = Delaunay(pixels).simplices
    image_edges = np.linalg.norm(
        pixels[faces] - pixels[np.roll(faces, 1, axis=1)], axis=2
    )
    metric_edges = np.linalg.norm(
        vertices[faces] - vertices[np.roll(faces, 1, axis=1)], axis=2
    )
    keep = (image_edges.max(1) <= maximum_pixel_edge) & (
        metric_edges.max(1) <= maximum_metric_edge_m
    )
    # Check the triangle interior and edges on a subpixel grid. A thin masked
    # obstacle must not be bridged just because the three vertices are visible.
    steps = max(2, int(np.ceil(maximum_pixel_edge * 2)))
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            pix = np.rint(
                (
                    pixels[faces[:, 0]] * a
                    + pixels[faces[:, 1]] * b
                    + pixels[faces[:, 2]] * (steps - a - b)
                )
                / steps
            ).astype(int)
            ok = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
            ids = np.flatnonzero(ok)
            ok[ids] &= mask[pix[ids, 1], pix[ids, 0]]
            keep &= ok
    faces = faces[keep]
    if not len(faces):
        raise ValueError("No locally supported triangles")
    used, remapped = np.unique(faces, return_inverse=True)
    faces = remapped.reshape(-1, 3)
    world = vertices[used] @ T[:3, :3].T + T[:3, 3]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(used), len(used))
    )
    count, labels = connected_components(graph, directed=False)
    return {
        "vertices": world,
        "faces": faces,
        "pixels": pixels[used],
        "cell_depth_range_m": np.array(ranges)[used],
        "cell_sample_count": np.array(counts)[used],
        "component_labels": labels,
        "components": int(count),
        "thickness_m": None,
    }
