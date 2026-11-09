import numpy as np
import pytest
import trimesh
from real2sim2real.controller_actor_training.planar_collision_base import propose


def test_noisy_base_clipping_is_bounded_and_preserves_tangential_geometry():
    mesh=trimesh.creation.box(extents=[.4,.3,.2])
    v=mesh.vertices.copy();v[0,2]-=.004
    mesh=trimesh.Trimesh(v,mesh.faces,process=False).convex_hull
    original=mesh.vertices.copy()
    vertices,faces,report=propose(mesh.vertices,mesh.faces,[0,0,1],cut_above_lowest_m=.006)
    np.testing.assert_array_equal(mesh.vertices,original)
    assert vertices[:,2].min()==pytest.approx(original[:,2].min()+.006)
    np.testing.assert_allclose(vertices[:,:2].min(0),original[:,:2].min(0))
    np.testing.assert_allclose(vertices[:,:2].max(0),original[:,:2].max(0))
    assert report['max_vertex_correction_m']<=.00600001
    assert report['proposal_only'] and not report['physics_validated']
    assert trimesh.Trimesh(vertices,faces,process=False).is_watertight


def test_rotated_support_normal_keeps_same_bounded_intervention():
    from scipy.spatial.transform import Rotation
    mesh=trimesh.creation.box(extents=[.4,.3,.2]);r=Rotation.from_euler('x',25,degrees=True).as_matrix()
    v=mesh.vertices@r.T;n=r[:,2]
    new,_,report=propose(v,mesh.faces,n,cut_above_lowest_m=.004)
    assert (new@n).min()==pytest.approx((v@n).min()+.004)
    assert report['volume_loss_fraction']==pytest.approx(.02)


@pytest.mark.parametrize('kwargs',[dict(cut_above_lowest_m=.02),dict(cut_above_lowest_m=-.001),
    dict(cut_above_lowest_m=.01,max_volume_loss_fraction=.001)])
def test_unbounded_or_large_volume_changes_are_rejected(kwargs):
    mesh=trimesh.creation.box(extents=[.4,.3,.2])
    with pytest.raises(ValueError):propose(mesh.vertices,mesh.faces,[0,0,1],**kwargs)
