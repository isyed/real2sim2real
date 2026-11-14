"""Cached CPU rendering of enabled convex collision surfaces, without stepping."""

import numpy as np


class CollisionRenderer:
    """Review geometry only; transparency and colors do not alter simulation."""

    def __init__(
        self,
        model,
        center_m,
        *,
        colors=None,
        transparent_geoms=(),
        hidden_geoms=(),
        static_surfaces=(),
    ):
        import mujoco
        import trimesh

        self.center = np.asarray(center_m, float)
        if self.center.shape != (3,) or not np.isfinite(self.center).all():
            raise ValueError("Finite metric camera center required")
        self.eye = self.center + np.array([0.38, -0.38, 0.31])
        z = self.center - self.eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        self.basis = np.column_stack([x, -np.cross(x, z), z])
        self.pieces = []
        # Optional observed triangles preserve holes/concavities for display.
        # They NEVER alter collision geometry, model parameters or simulation.
        # Reports using them must label this as a surface view, not a collider view.
        self.static_surfaces = []
        for triangles, color in static_surfaces:
            triangles = np.array(triangles, float, copy=True)
            color = np.array(color, float, copy=True)
            if (
                triangles.ndim != 3
                or triangles.shape[1:] != (3, 3)
                or not len(triangles)
                or not np.isfinite(triangles).all()
                or color.shape != (3,)
                or not np.isfinite(color).all()
                or (color < 0).any()
                or (color > 255).any()
            ):
                raise ValueError("Finite world-space triangles and RGB color required")
            self.static_surfaces.append((triangles, color))
        known = {model.geom(g).name for g in range(model.ngeom)}
        if set(hidden_geoms) - known:
            raise ValueError("Unknown hidden geom name")
        hidden_geoms = set(hidden_geoms)
        cache = {}
        colors = colors or {}
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            name = model.geom(g).name
            if name in hidden_geoms:
                continue
            kind = model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                if mid not in cache:
                    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                    # MuJoCo's ordinary mesh contact geometry is the convex hull.
                    cache[mid] = trimesh.convex.convex_hull(
                        model.mesh_vert[start : start + count]
                    ).triangles.copy()
                triangles = cache[mid]
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                triangles = trimesh.creation.icosphere(
                    subdivisions=2, radius=model.geom_size[g, 0]
                ).triangles
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                triangles = trimesh.creation.cylinder(
                    radius=model.geom_size[g, 0],
                    height=model.geom_size[g, 1] * 2,
                    sections=32,
                ).triangles
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                triangles = trimesh.creation.box(
                    extents=model.geom_size[g] * 2
                ).triangles
            else:
                raise ValueError(f"Unsupported review geom type: {kind}")
            self.pieces.append(
                (
                    g,
                    triangles,
                    np.array(colors.get(name, [65, 145, 205])),
                    70 if name in transparent_geoms else 255,
                )
            )

    def frame(self, data):
        from PIL import Image, ImageDraw

        triangles = []
        colors = []
        alphas = []
        for surface, color in self.static_surfaces:
            triangles.append(surface)
            colors.append(np.tile(color, (len(surface), 1)))
            alphas.append(np.full(len(surface), 255))
        for gid, local, color, alpha in self.pieces:
            triangles.append(
                local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            )
            colors.append(np.tile(color, (len(local), 1)))
            alphas.append(np.full(len(local), alpha))
        t = np.concatenate(triangles)
        colors = np.concatenate(colors)
        alphas = np.concatenate(alphas)
        camera = (t - self.eye) @ self.basis
        xy = camera[:, :, :2] / camera[:, :, 2:] * 850 + [450, 320]
        normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        shade = np.clip(
            colors * (0.5 + 0.5 * abs(normal @ np.array([0.3, -0.5, 0.812])))[:, None],
            0,
            255,
        ).astype(np.uint8)
        rgba = np.column_stack([shade, alphas]).tolist()
        polygons = xy.reshape(-1, 6).tolist()
        im = Image.new("RGB", (900, 640), (245, 245, 245))
        draw = ImageDraw.Draw(im, "RGBA")
        for i in np.argsort(-camera[:, :, 2].mean(1)):
            if (camera[i, :, 2] > 0.01).all():
                draw.polygon(polygons[i], fill=tuple(rgba[i]))
        return im
