import numpy as np
import trimesh

from real2sim2real.egocentric.visible_region_geometry import image_regions, ray_depth


def test_regions_keep_hole_and_background_separate_and_exclude_hands():
    mask = np.zeros((80, 80), bool)
    mask[10:65, 20:60] = True
    mask[45:55, 35:45] = False
    occluder = np.zeros_like(mask)
    occluder[55:, 20:32] = True
    regions = image_regions(mask, occluder, lower_fraction=0.5)
    assert regions["hole"][50, 40] and not regions["focus"][50, 40]
    assert not any(m[60, 25] for m in regions.values())
    assert regions["nearby_table"][65, 65]
    assert not any(
        np.any(a & b)
        for i, a in enumerate(regions.values())
        for b in list(regions.values())[i + 1 :]
    )


def test_ray_depth_uses_visible_surface_and_preserves_missing_rays():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    pose = np.eye(4)
    pose[2, 3] = 1
    k = np.diag([100, 100, 1])
    uv = np.array([[0, 0], [30, 0]])
    depth = ray_depth(mesh, pose, uv, k)
    np.testing.assert_allclose(depth[0], 0.95)
    assert np.isnan(depth[1])
