"""Small CPU geometry operations in explicit world/camera coordinates."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp


def mesh_center(g: dict) -> np.ndarray:
    triangle = g["vertices"][g["faces"]]
    valid = np.any(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]) != 0, axis=1)
    vertices = g["vertices"][np.unique(g["faces"][valid])] if valid.any() else g["vertices"]
    return (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5 * g["object_scale"]


def object_centers(g: dict) -> np.ndarray:
    return np.einsum("tij,j->ti", g["object_rotation"], mesh_center(g)) + g["object_translation"]


def object_world_vertices(g: dict, i: int) -> np.ndarray:
    return g["vertices"] * g["object_scale"] @ g["object_rotation"][i].T + g["object_translation"][i]


def object_camera_centers(g: dict) -> np.ndarray:
    w = g["world_to_camera"]
    return np.einsum("tij,tj->ti", w[:, :3, :3], object_centers(g)) + w[:, :3, 3]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation mapping direction a to b, including antiparallel directions."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross, dot = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    if np.linalg.norm(cross) < 1e-10:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.arccos(dot)).as_matrix()


def rotation_interpolate(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    return Slerp([0, 1], Rotation.from_matrix(np.stack([a, b])))([fraction]).as_matrix()[0]


def rotation_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(a @ b.swapaxes(-1, -2)).magnitude()


def mesh_silhouette(g: dict, i: int, height: int, width: int,
                    intrinsics: np.ndarray | None = None) -> np.ndarray:
    """Triangle union, not a convex hull: holes in the input surface survive.

    Geometry crossing/behind the near plane is withheld by scoring; we do not
    invent clipped triangles or depth. Camera intrinsics match this pixel grid.
    """
    w = g["world_to_camera"][i]
    camera = object_world_vertices(g, i) @ w[:3, :3].T + w[:3, 3]
    k = g["intrinsics"][i] if intrinsics is None else intrinsics
    projection = camera @ k.T
    pixels = projection[:, :2] / np.maximum(projection[:, 2:], 1e-8)
    faces = g["faces"]
    triangles = pixels[faces]
    valid = (camera[faces, 2] > 1e-5).all(axis=1)
    valid &= (triangles[:, :, 0].max(axis=1) >= 0) & (triangles[:, :, 0].min(axis=1) < width)
    valid &= (triangles[:, :, 1].max(axis=1) >= 0) & (triangles[:, :, 1].min(axis=1) < height)
    cross = np.cross(g["vertices"][faces[:, 1]] - g["vertices"][faces[:, 0]],
                     g["vertices"][faces[:, 2]] - g["vertices"][faces[:, 0]])
    valid &= np.linalg.norm(cross, axis=1) > 1e-14
    canvas = Image.new("1", (width, height))
    draw = ImageDraw.Draw(canvas)
    raster_triangles = triangles[valid]
    if len(raster_triangles) > 20000:
        # Pillow truncates polygon coordinates to integers. Dense predicted
        # meshes often repeat the same raster triangle thousands of times.
        # Removing identical integer polygons preserves their exact union;
        # no geometry, topology, depth or silhouette resolution is changed.
        raster_triangles = np.unique(np.trunc(raster_triangles).astype(np.int64).reshape(-1, 6), axis=0).reshape(-1, 3, 2)
    for triangle in raster_triangles:
        draw.polygon([tuple(p) for p in triangle], fill=1)
    return np.asarray(canvas, dtype=bool)


def frame_overlap(g: dict, o: dict, i: int, max_size: int | None = None, *, crop_to_object: bool = False) -> float:
    target = o["object_mask"][i].astype(bool)
    occluder = o.get("occluder_mask")
    occ = np.zeros_like(target) if occluder is None else occluder[i].astype(bool)
    h, w = target.shape
    if crop_to_object:
        # Crop the union AFTER full-grid rasterization, so projected pixels
        # outside the observed target remain part of the objective. Never crop
        # to the target alone or silently move/rescale camera intrinsics.
        render = mesh_silhouette(g, i, h, w) & ~occ
        target &= ~occ
        y, x = np.where(render | target)
        if not len(x):
            return 0.0
        top, bottom = max(0, y.min()-2), min(h, y.max()+3)
        left, right = max(0, x.min()-2), min(w, x.max()+3)
        render, target = render[top:bottom, left:right], target[top:bottom, left:right]
        h, w = target.shape
        if max_size is not None and max(h, w) > max_size:
            scale = max_size / max(h, w)
            size = (max(2, round(w*scale)), max(2, round(h*scale)))
            render = np.asarray(Image.fromarray(render).resize(size, Image.Resampling.NEAREST))
            target = np.asarray(Image.fromarray(target).resize(size, Image.Resampling.NEAREST))
        union = np.count_nonzero(render | target)
        return float(np.count_nonzero(render & target) / union) if union else 0.0
    k = g["intrinsics"][i].copy()
    if max_size is not None and max(h, w) > max_size:
        scale = max_size / max(h, w)
        nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
        target = np.asarray(Image.fromarray(target).resize((nw, nh), Image.Resampling.NEAREST))
        occ = np.asarray(Image.fromarray(occ).resize((nw, nh), Image.Resampling.NEAREST))
        k[0] *= nw / w
        k[1] *= nh / h
        h, w = nh, nw
    render = mesh_silhouette(g, i, h, w, k) & ~occ
    target = target & ~occ
    union = np.count_nonzero(render | target)
    return float(np.count_nonzero(render & target) / union) if union else 0.0


def mesh_audit(g: dict) -> dict:
    v, f = g["vertices"], g["faces"]
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    valid = np.linalg.norm(cross, axis=1) > 1e-14
    real = f[valid]
    edges = np.concatenate([real[:, [0, 1]], real[:, [1, 2]], real[:, [2, 0]]])
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    used = np.unique(real)
    duplicate_faces = len(real) - len(np.unique(np.sort(real, axis=1), axis=0))
    return {
        "vertices": int(len(v)), "faces": int(len(f)), "used_vertices": int(len(used)),
        "degenerate_faces": int(np.count_nonzero(~valid)), "duplicate_faces": duplicate_faces,
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "euler_characteristic": int(len(used) - len(unique) + len(real)),
        "local_extent_m": (np.ptp(v[used], axis=0) * g["object_scale"]).tolist() if len(used) else [0, 0, 0],
    }


def mesh_surface_equal(a: dict, b: dict) -> bool:
    """Compare exact oriented triangle sets, independent of padding/indexing."""
    def triangles(geometry):
        v = geometry["vertices"][geometry["faces"]]
        v = v[np.any(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]) != 0, axis=1)]
        return {min(tuple(np.roll(t, shift, axis=0).reshape(-1)) for shift in range(3)) for t in v}
    return triangles(a) == triangles(b)


def points_mesh_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Exact unsigned point-to-triangle distance; not penetration or hand contact GT."""
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    valid = norm2 > 1e-24
    # Exact broad phase for dense meshes. The existing inside test permits
    # barycentric coordinates down to -1e-14 / ||normal||^2. Bound that SAME
    # expanded triangle (rather than silently tightening its tolerance).
    # A vertex is an upper bound on surface distance. Faces whose expanded
    # box is farther cannot improve that bound; their edge/plane work is skipped.
    dense = len(tri) > 2048
    if dense:
        low, high = tri.min(axis=1), tri.max(axis=1)
        epsilon = np.where(valid, 1e-14 / np.maximum(norm2, 1e-30), 0.)
        summed = tri.sum(axis=1)
        box_low = low - epsilon[:, None] * (summed - 3*low)
        box_high = high + epsilon[:, None] * (3*high - summed)
    distances = []
    for p in points:
        if dense:
            upper = np.min(np.einsum('ij,ij->i', p-a, p-a))
            delta = np.maximum(np.maximum(box_low-p, p-box_high), 0.)
            lower = np.einsum('ij,ij->i', delta, delta)
            keep = lower <= upper + 1e-12 * max(1., float(upper))
            aa, bb, cc = a[keep], b[keep], c[keep]
            nn, n2, vv = normal[keep], norm2[keep], valid[keep]
        else:
            aa, bb, cc, nn, n2, vv = a, b, c, normal, norm2, valid
        ap = p - aa
        signed = np.einsum("ij,ij->i", ap, nn)
        projection = p - nn * (signed / np.maximum(n2, 1e-30))[:, None]
        inside = vv.copy()
        candidates = []
        for u, v in ((aa, bb), (bb, cc), (cc, aa)):
            edge = v - u
            inside &= np.einsum("ij,ij->i", np.cross(edge, projection - u), nn) >= -1e-14
            t = np.clip(np.einsum("ij,ij->i", p - u, edge) /
                        np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30), 0, 1)
            candidates.append(np.linalg.norm(p - (u + t[:, None] * edge), axis=1))
        distances.append(min(float(np.min(np.stack(candidates))),
                             float(np.min(np.where(inside, np.abs(signed) / np.sqrt(np.maximum(n2, 1e-30)), np.inf)))))
    return np.asarray(distances)


def reliable_frames(o: dict, n: int) -> np.ndarray:
    if "object_mask" not in o:
        return np.zeros(n, dtype=bool)
    mask = o["object_mask"].astype(bool)
    if "occluder_mask" in o:
        mask = mask & ~o["occluder_mask"].astype(bool)
    return o["reliable"].astype(bool) & mask.any(axis=(1, 2))


def valid_segments(g: dict, o: dict, max_gap_s: float) -> list[np.ndarray]:
    valid = reliable_frames(o, len(g["frame_ids"])) & ~o.get("locked", np.zeros(len(g["frame_ids"]), bool)).astype(bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    split = (np.diff(ids) > 1) | (np.diff(g["timestamps_s"][ids]) > max_gap_s)
    # An annotated contact transition is a boundary, not a smoothing target.
    if "contact_active" in o:
        split |= np.any(o["contact_active"][ids[1:]] != o["contact_active"][ids[:-1]], axis=1)
    return list(np.split(ids, np.flatnonzero(split) + 1))
