"""Build explicit static collision surfaces from selected predicted depth.

ROI selection and coordinate alignment belong to the caller. This method never
loads a ground-truth scene or invents thickness for a degenerate surface.
"""
import numpy as np

def static_geometry(depth, K, human, obj, rectangles, floor, *, polygons=(), mode='surface'):
    """Recover observed support surfaces by default, with explicit legacy opt-in.

    Broad rectangular selections must not become filled scene-sized hulls. Use
    segmented_support for explicit furniture parts; the default recovers only
    an observed horizontal surface and reports missing vertical structure.
    """
    if mode=='surface':
        from .surface_support import planar_region_geometry
        return planar_region_geometry(depth,K,human,obj,rectangles,floor,polygons=polygons)
    if mode!='legacy_convex':raise ValueError('Unknown support recovery mode')
    return _legacy_geometry(depth,K,human,obj,rectangles,floor,polygons=polygons)


def _legacy_geometry(depth, K, human, obj, rectangles, floor, *, polygons=()):
    """Backproject only selected, unoccluded predicted static pixels.

    A convex hull is an explicit collision approximation to the selected depth
    surface. No clean table model, dimensions or motion are supplied.
    """
    from scipy.ndimage import binary_dilation
    from scipy.spatial import ConvexHull
    keep = np.zeros(depth.shape, bool)
    for x0, y0, x1, y1 in rectangles:
        if not (0 <= x0 < x1 <= depth.shape[1] and 0 <= y0 < y1 <= depth.shape[0]):
            raise ValueError('Static support ROI is outside the RGB frame')
        keep[y0:y1, x0:x1] = True
    if polygons:
        from PIL import Image, ImageDraw
        mask = Image.new('1', (depth.shape[1], depth.shape[0]))
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            xy = np.asarray(polygon)
            if (xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) < 3
                    or not np.isfinite(xy).all() or np.any(xy < 0)
                    or np.any(xy[:, 0] >= depth.shape[1]) or np.any(xy[:, 1] >= depth.shape[0])):
                raise ValueError('Static support polygon is outside the RGB frame')
            draw.polygon([tuple(p) for p in xy.tolist()], fill=1)
        keep |= np.asarray(mask, dtype=bool)
    keep &= ~binary_dilation(human | obj, iterations=4)
    keep &= np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    v, u = np.nonzero(keep)
    if len(u) < 300:
        raise ValueError('Insufficient visible predicted support depth; do not omit support')
    # Deterministic spatial subsampling, independent of clean scores.
    ids = np.linspace(0, len(u)-1, min(len(u), 6000), dtype=int)
    rays = np.column_stack((u[ids], v[ids], np.ones(len(ids)))) @ np.linalg.inv(K).T
    points = rays * depth[v[ids], u[ids], None]
    height = points @ floor[:3] + floor[3]
    points = points[(height > .02) & (height < 2.0)]
    if len(points) < 300:
        raise ValueError('Selected support depth does not rise above the predicted floor')
    median = np.median(points, axis=0)
    radius = np.linalg.norm(points-median, axis=1)
    points = points[radius <= np.percentile(radius, 98)]
    if np.linalg.svd(points-points.mean(0), compute_uv=False)[-1] < .001:
        raise ValueError('Predicted support is degenerate; do not invent a thickness')
    hull = ConvexHull(points)
    span = np.ptp(points, axis=0)
    if not .00001 < hull.volume < 8 or np.max(span) > 4:
        raise ValueError('Predicted support hull has implausible dimensions: '
                         f'volume={hull.volume:.6g} m3, spans={span.tolist()} m')
    # Face orientation is needed by static mesh importers.
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = points[face]
        if np.dot(np.cross(b-a, c-a), hull.equations[i, :3]) < 0:
            faces[i] = face[::-1]
    return points, faces, dict(visible_pixels=len(u), hull_volume_m3=float(hull.volume),
        collision_approximation='convex hull of RGB-selected predicted depth; no fitted primitive')


def write_support(path, meshes):
    text = '#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n)\ndef Xform "World" {\n'
    for index, (vertices, faces) in enumerate(meshes):
        pts = ', '.join('('+', '.join(format(float(x), '.9g') for x in p)+')' for p in vertices)
        counts = ', '.join('3' for _ in faces)
        indices = ', '.join(str(int(x)) for x in np.asarray(faces).ravel())
        text += (f'    def Mesh "Support{index}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{\n'
                 f'        point3f[] points = [{pts}]\n'
                 f'        int[] faceVertexCounts = [{counts}]\n'
                 f'        int[] faceVertexIndices = [{indices}]\n'
                 '        uniform token subdivisionScheme = "none"\n'
                 '        bool physics:collisionEnabled = true\n'
                 '    }\n')
    path.write_text(text+'}\n')

