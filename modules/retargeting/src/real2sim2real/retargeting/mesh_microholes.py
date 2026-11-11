"""Explicit small boundary repair; never seal large openings or alter vertices."""

import numpy as np


def close_microholes(mesh, *, max_diameter_m=0.001, max_edges=4):
    if not 0 < max_diameter_m <= 0.002 or max_edges not in (3, 4):
        raise ValueError(
            "Only explicitly bounded sub-2mm triangle/quad repairs allowed"
        )
    if not mesh.is_winding_consistent:
        raise ValueError("Consistent source winding required")
    unique, inverse, counts = np.unique(
        np.sort(mesh.edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError("Nonmanifold source edges cannot be repaired here")
    directed = mesh.edges[counts[inverse] == 1]
    edges = {int(a): int(b) for a, b in directed}
    if len(edges) != len(directed) or len(set(edges.values())) != len(edges):
        raise ValueError("Boundary must consist of disjoint simple cycles")
    faces, repairs = [], []
    while edges:
        start = next(iter(edges))
        loop = []
        current = start
        while True:
            loop.append(current)
            if current not in edges or len(loop) > max_edges:
                raise ValueError("Boundary exceeds the small-hole repair budget")
            current = edges.pop(current)
            if current == start:
                break
        points = mesh.vertices[loop]
        diameter = np.linalg.norm(points[:, None] - points[None], axis=-1).max()
        if len(loop) < 3 or diameter > max_diameter_m:
            raise ValueError("Boundary exceeds the metric repair budget")
        # Reverse source boundary winding; preserve all source vertices/faces.
        faces.extend([[loop[0], loop[j + 1], loop[j]] for j in range(1, len(loop) - 1)])
        repairs.append(dict(vertices=loop, diameter_m=float(diameter)))
    result = mesh.copy()
    if faces:
        result.faces = np.vstack([mesh.faces, faces])
    if (
        not result.is_watertight
        or not result.is_winding_consistent
        or result.volume <= 0
    ):
        raise ValueError("Bounded repair did not produce a valid closed surface")
    return result, dict(
        repairs=repairs,
        faces_added=len(faces),
        vertices_changed=False,
        max_diameter_m=max_diameter_m,
        source_watertight=bool(mesh.is_watertight),
        accepted=True,
        contact_geometry_validated=False,
    )
