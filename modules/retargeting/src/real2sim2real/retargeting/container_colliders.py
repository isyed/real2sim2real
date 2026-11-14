"""Explicit hollow tapered-container collision hypotheses in metres.

Each returned piece is convex, so simulators do not fill the container opening
when taking per-mesh convex hulls. Thickness is supplied, never inferred here.
"""

import numpy as np
from scipy.spatial import ConvexHull


def _convex_piece(name, vertices):
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    return {
        "name": name,
        "vertices_m": vertices,
        "faces": faces,
        "volume_m3": float(hull.volume),
    }


def tapered_shell(
    bottom_radius_m, rim_radius_m, height_m, wall_m, base_m, *, sectors=48
):
    """Return closed convex pieces; ``wall_m`` is a radial thickness.

    Origin is the bottom centre, with the cup axis along +Z. Adjacent walls
    share faces, and the base overlaps their bottom ends. Piece volumes must
    not be summed to infer mass or inertia without accounting for that overlap.
    """
    values = np.asarray(
        [bottom_radius_m, rim_radius_m, height_m, wall_m, base_m], float
    )
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or wall_m >= min(bottom_radius_m, rim_radius_m) / 2
        or base_m >= height_m / 4
        or isinstance(sectors, bool)
        or not isinstance(sectors, int)
        or not 16 <= sectors <= 128
    ):
        raise ValueError(
            "Positive metric shell dimensions and bounded sector count required"
        )
    angles = np.linspace(0, 2 * np.pi, sectors + 1)
    pieces = []
    for i in range(sectors):
        vertices = np.array(
            [
                [r * np.cos(a), r * np.sin(a), z]
                for z, outer in [(0, bottom_radius_m), (height_m, rim_radius_m)]
                for r in [outer, outer - wall_m]
                for a in angles[i : i + 2]
            ]
        )
        pieces.append(_convex_piece(f"wall_{i:03d}", vertices))
    top_radius = bottom_radius_m + (rim_radius_m - bottom_radius_m) * base_m / height_m
    vertices = np.array(
        [
            [r * np.cos(a), r * np.sin(a), z]
            for z, r in [(0, bottom_radius_m), (base_m, top_radius)]
            for a in angles[:-1]
        ]
    )
    pieces.append(_convex_piece("base", vertices))
    return pieces
