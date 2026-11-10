"""Opening discovery and conservative swept-sphere clearance in a metric world."""

import numpy as np
from scipy import ndimage
import trimesh


def vertical_grid(mesh, *, pitch_m=0.001, padding_m=0.01, maximum_cells=100000):
    """Sample projected solid columns; enclosed empty components suggest openings.

    The raster is a discovery diagnostic, not a clearance certificate. Thin
    triangles can fall between rays; verify any proposed path against triangles.
    """
    if (
        not np.isfinite([pitch_m, padding_m]).all()
        or pitch_m <= 0
        or padding_m < pitch_m
    ):
        raise ValueError("Finite positive metric grid and padding required")
    lo, hi = mesh.bounds
    x = np.arange(lo[0] - padding_m, hi[0] + padding_m + pitch_m, pitch_m)
    y = np.arange(lo[1] - padding_m, hi[1] + padding_m + pitch_m, pitch_m)
    if len(x) * len(y) > maximum_cells:
        raise ValueError("Opening grid exceeds cell budget")
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[xx.ravel(), yy.ravel()]
    low = np.full(len(xy), np.inf)
    high = np.full(len(xy), -np.inf)
    for start in range(0, len(xy), 1024):
        origins = np.c_[
            xy[start : start + 1024],
            np.full(len(xy[start : start + 1024]), lo[2] - 0.05),
        ]
        hit, rays, _ = mesh.ray.intersects_location(
            origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)), multiple_hits=True
        )
        if len(hit):
            np.minimum.at(low, start + rays, hit[:, 2])
            np.maximum.at(high, start + rays, hit[:, 2])
    occupied = np.isfinite(low).reshape(xx.shape)
    holes = ndimage.binary_fill_holes(occupied) & ~occupied
    labels, count = ndimage.label(holes)
    distance = ndimage.distance_transform_edt(~occupied) * pitch_m
    openings = []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        best = cells[np.argmax(distance[labels == label])]
        row, col = best
        openings.append(
            dict(
                component=label,
                area_m2=float(len(cells) * pitch_m**2),
                center_xy_m=[float(x[col]), float(y[row])],
                raster_inscribed_radius_m=float(distance[row, col]),
                raster_error_allowance_m=float(np.sqrt(2) * pitch_m),
                cell_count=len(cells),
            )
        )
    openings.sort(key=lambda r: -r["area_m2"])
    return dict(
        x=x,
        y=y,
        occupied=occupied,
        low=low.reshape(xx.shape),
        high=high.reshape(xx.shape),
        labels=labels,
        distance=distance,
        openings=openings,
        pitch_m=pitch_m,
    )


def union_columns(convex_pieces, xy):
    """Exact line/half-space intersection for the union's projected occupancy."""
    xy = np.asarray(xy, float)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or not np.isfinite(xy).all()
        or not convex_pieces
    ):
        raise ValueError("Finite XY positions and convex pieces required")
    if any(
        not m.is_watertight or not m.is_convex or m.volume <= 0 for m in convex_pieces
    ):
        raise ValueError("Closed outward convex pieces required")
    occupied = np.zeros(len(xy), bool)
    for mesh in convex_pieces:
        normals = mesh.face_normals
        points = mesh.triangles[:, 0]
        # Ignore pieces whose XY bounds cannot contain the query.
        ids = np.flatnonzero(
            np.all(
                (xy >= mesh.bounds[0, :2] - 1e-8) & (xy <= mesh.bounds[1, :2] + 1e-8),
                axis=1,
            )
        )
        if not len(ids):
            continue
        b = xy[ids] @ normals[:, :2].T - np.einsum("ij,ij->i", normals, points)
        slope = normals[:, 2]
        parallel = abs(slope) < 1e-10
        feasible = ~(b[:, parallel] > 1e-8).any(1)
        upper = np.min(-b[:, slope > 1e-10] / slope[slope > 1e-10], axis=1)
        lower = np.max(-b[:, slope < -1e-10] / slope[slope < -1e-10], axis=1)
        occupied[ids] |= feasible & (upper >= lower)
    return occupied


def polyline_samples(vertices, maximum_step_m=0.001):
    vertices = np.asarray(vertices, float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 2
        or not np.isfinite(vertices).all()
        or not np.isfinite(maximum_step_m)
        or maximum_step_m <= 0
    ):
        raise ValueError("Finite metric polyline and positive sampling step required")
    samples = []
    bound = 0.0
    for a, b in zip(vertices[:-1], vertices[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(length / maximum_step_m)))
        if n > 10000:
            raise ValueError("Segment sampling budget exceeded")
        samples.extend(np.linspace(a, b, n + 1))
        bound = max(bound, length / n / 2)
    if len(samples) > 50000:
        raise ValueError("Path sampling budget exceeded")
    return np.asarray(samples), bound


def signed_clearance(mesh, points, *, convex_pieces=None):
    """Distance to the mesh boundary, negative inside a solid or convex union."""
    points = np.asarray(points, float)
    distances = []
    inside = []
    for start in range(0, len(points), 256):
        part = points[start : start + 256]
        distances.extend(trimesh.proximity.closest_point(mesh, part)[1])
        if convex_pieces is None:
            inside.extend(mesh.contains(part))
        else:
            occupied = np.zeros(len(part), bool)
            for piece in convex_pieces:
                ids = np.flatnonzero(
                    np.all(
                        (part >= piece.bounds[0] - 1e-8)
                        & (part <= piece.bounds[1] + 1e-8),
                        axis=1,
                    )
                )
                if len(ids):
                    n = piece.face_normals
                    offsets = np.einsum("ij,ij->i", n, piece.triangles[:, 0])
                    occupied[ids] |= np.all(part[ids] @ n.T - offsets <= 1e-8, axis=1)
            inside.extend(occupied)
    return np.asarray(distances) * np.where(inside, -1.0, 1.0)


def sweep_clearance(
    obstacles, centerlines, *, radius_m, margin_m, maximum_step_m=0.001
):
    """Bound a union of swept balls against explicit meshes, with path witnesses.

    Each polyline describes a ball-center trace. If these traces represent a
    discretized rigid tool, the caller must add its discretization cover radius
    to margin_m. Positive bounds certify this envelope only, not a whole robot.
    """
    if (
        not np.isfinite([radius_m, margin_m]).all()
        or radius_m <= 0
        or margin_m < 0
        or not centerlines
        or not obstacles
    ):
        raise ValueError(
            "Explicit obstacles, traces, positive radius and margin required"
        )
    samples = []
    cover = 0.0
    for line in centerlines:
        p, bound = polyline_samples(line, maximum_step_m)
        samples.extend(p)
        cover = max(cover, bound)
    samples = np.asarray(samples)
    if len(samples) > 50000:
        raise ValueError("Sweep sampling budget exceeded")
    rows = []
    for name, mesh, pieces in obstacles:
        distances = signed_clearance(mesh, samples, convex_pieces=pieces)
        index = int(np.argmin(distances))
        raw = float(distances[index] - radius_m)
        lower = raw - cover - margin_m
        rows.append(
            dict(
                obstacle=name,
                sampled_clearance_m=raw,
                conservative_clearance_m=lower,
                witness_center_m=samples[index].tolist(),
                accepted=lower >= 0,
            )
        )
    return dict(
        accepted=all(r["accepted"] for r in rows),
        obstacles=rows,
        sampled_centers=len(samples),
        centerline_cover_radius_m=cover,
        radius_m=radius_m,
        margin_m=margin_m,
        whole_robot_validated=False,
        physical_success_validated=False,
    )
