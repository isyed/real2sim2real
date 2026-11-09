"""Deterministic surface registration with fixed physical trajectory scale."""

from itertools import permutations, product
import numpy as np
from scipy.spatial import cKDTree


def sample_surface(mesh, count=3000, seed=0):
    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles, float)
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    if not np.isfinite(triangles).all() or not areas.sum() > 0:
        raise ValueError("Mesh has no finite positive-area surface")
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1 - uv[flip]
    return (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    )


def fit_rigid(source, target):
    a = np.asarray(source, float)
    b = np.asarray(target, float)
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, cb - rotation @ ca


def register(candidate, reference, *, samples=3000, seed=0):
    a = sample_surface(candidate, samples, seed)
    b = sample_surface(reference, samples, seed + 1)
    ca, cb = a.mean(0), b.mean(0)
    scale = float(
        np.sqrt(((b - cb) ** 2).sum(1).mean() / ((a - ca) ** 2).sum(1).mean())
    )
    aa = (a - ca) * scale
    bb = b - cb
    _, axes_a = np.linalg.eigh(aa.T @ aa)
    _, axes_b = np.linalg.eigh(bb.T @ bb)
    tree = cKDTree(bb)
    best = None
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            r = axes_b @ np.eye(3)[:, order] @ np.diag(signs) @ axes_a.T
            if np.linalg.det(r) < 0:
                continue
            offset = np.zeros(3)
            for _ in range(35):
                moved = aa @ r.T + offset
                dist, indices = tree.query(moved)
                keep = dist <= np.quantile(dist, 0.9)
                delta, shift = fit_rigid(moved[keep], bb[indices[keep]])
                r = delta @ r
                offset = delta @ offset + shift
                if (
                    np.linalg.norm(shift) < 1e-6
                    and np.linalg.norm(delta - np.eye(3)) < 1e-5
                ):
                    break
            moved = aa @ r.T + offset
            distance = float(
                (tree.query(moved)[0].mean() + cKDTree(moved).query(bb)[0].mean()) / 2
            )
            if best is None or distance < best[0]:
                best = (distance, r, offset)
    distance, r, offset = best
    return dict(
        chamfer_m=distance,
        scale=scale,
        rotation=r,
        translation=cb + offset - scale * r @ ca,
        candidate_center=ca,
        reference_center=cb,
        method="surface_pca_multistart_trimmed_icp",
    )
