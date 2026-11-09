"""Exact point-to-triangle distances with bounded candidate batches.

Triangle centroids are partitioned by bounding radius. A face can improve an
upper distance bound d only if its centroid is within d + its bounding radius.
This excludes distant faces without the large axis-aligned query boxes produced
by generic proximity routines for points far outside a dense mesh.
"""

import numpy as np
from scipy.spatial import cKDTree


class MeshSurfaceQuery:
    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles, dtype=np.float64)
        if not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError("Require a finite nonempty triangle surface")
        self.centres = self.triangles.mean(1)
        self.radii = np.linalg.norm(self.triangles - self.centres[:, None], axis=2).max(
            1
        )
        self.tree = cKDTree(self.centres)
        bins = np.floor(np.log2(np.maximum(self.radii, 1e-12))).astype(int)
        self.groups = []
        for b in np.unique(bins):
            indices = np.flatnonzero(bins == b)
            self.groups.append(
                (
                    indices,
                    cKDTree(self.centres[indices]),
                    float(self.radii[indices].max()),
                )
            )

    def closest(self, points):
        from trimesh.triangles import closest_point

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite Nx3 query points")
        count = len(points)
        if not count:
            return points.copy(), np.empty(0), np.empty(0, dtype=np.int64)
        # An exact projection onto nearby-centroid faces provides an upper bound,
        # not an approximation to the final answer.
        k = min(4, len(self.triangles))
        seeds = self.tree.query(points, k=k)[1].reshape(count, k)
        near = closest_point(
            self.triangles[seeds.ravel()], np.repeat(points, k, axis=0)
        ).reshape(count, k, 3)
        d2 = ((near - points[:, None]) ** 2).sum(2)
        chosen = d2.argmin(1)
        best2 = d2[np.arange(count), chosen].copy()
        best = near[np.arange(count), chosen].copy()
        faces = seeds[np.arange(count), chosen].copy()
        for indices, tree, radius in self.groups:
            candidates = tree.query_ball_point(points, np.sqrt(best2) + radius + 1e-12)
            lengths = np.array([len(c) for c in candidates])
            if not lengths.sum():
                continue
            point_ids = np.repeat(np.arange(count), lengths)
            face_ids = indices[np.concatenate(candidates).astype(np.int64)]
            # Per-face bounds further reduce candidates within each radius bin.
            bound = np.sqrt(best2[point_ids]) + self.radii[face_ids] + 1e-12
            keep = ((self.centres[face_ids] - points[point_ids]) ** 2).sum(
                1
            ) <= bound**2
            point_ids, face_ids = point_ids[keep], face_ids[keep]
            for start in range(0, len(face_ids), 32768):
                ids, face = (
                    point_ids[start : start + 32768],
                    face_ids[start : start + 32768],
                )
                projected = closest_point(self.triangles[face], points[ids])
                distance2 = ((projected - points[ids]) ** 2).sum(1)
                chunk_min = np.full(count, np.inf)
                np.minimum.at(chunk_min, ids, distance2)
                wins = (distance2 == chunk_min[ids]) & (distance2 < best2[ids])
                best2[ids[wins]] = distance2[wins]
                best[ids[wins]] = projected[wins]
                faces[ids[wins]] = face[wins]
        return best, np.sqrt(best2), faces
