"""Explicit finite-thickness triangle patches without whole-mesh convex filling."""

import numpy as np


def triangle_prisms(vertices_m, faces, *, thickness_m=0.004):
    """Return independent convex prisms below each upward-oriented triangle.

    Thickness is a hypothesis. Adjacent sharp faces can overlap or leave seams;
    this is a diagnostic collider representation, not a certified watertight solid.
    """
    v = np.asarray(vertices_m, float)
    f = np.asarray(faces)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1] != 3
        or not np.issubdtype(f.dtype, np.integer)
        or not len(f)
        or f.min() < 0
        or f.max() >= len(v)
        or not np.isfinite(thickness_m)
        or not 0 < thickness_m <= 0.02
    ):
        raise ValueError("Finite metric triangle mesh and bounded thickness required")
    top = v[f]
    normal = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    length = np.linalg.norm(normal, axis=1)
    if (length < 1e-12).any():
        raise ValueError("Degenerate bearing triangle")
    normal /= length[:, None]
    normal[normal[:, 2] < 0] *= -1
    return np.concatenate([top, top - thickness_m * normal[:, None]], axis=1)


def append_bearing_colliders(
    tree, vertices_m, faces, *, prefix="bearing_patch", thickness_m=0.004
):
    """Add a separate mesh per triangle; retain every existing obstacle."""
    import xml.etree.ElementTree as ET

    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Worldbody required")
    if any(x.get("name", "").startswith(prefix) for x in root.iter()):
        raise ValueError("Collider prefix already used")
    prisms = triangle_prisms(vertices_m, faces, thickness_m=thickness_m)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for i, p in enumerate(prisms):
        name = f"{prefix}_{i}"
        centre = p.mean(0)
        ET.SubElement(
            asset, "mesh", name=name, vertex=" ".join(map(str, (p - centre).ravel()))
        )
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            pos=" ".join(map(str, centre)),
            rgba=".8 .65 .3 1",
            contype="1",
            conaffinity="1",
            friction=".6 .005 .0001",
        )
    return {
        "added_patches": len(prisms),
        "thickness_m": thickness_m,
        "existing_geometries_retained": True,
        "watertight_verified": False,
    }
