import numpy as np
import pytest
from real2sim2real.retargeting.planar_outline import slab


def inputs():
    K=np.array([[1000.,0,500],[0,1000.,500],[0,0,1]])
    polygon=np.array([[300.,300],[700,300],[700,700],[300,700]])
    plane=np.array([0.,0,-1,2.])
    rng=np.random.default_rng(42)
    points=np.c_[rng.uniform(-.4,.4,(100,2)),np.full(100,2.)]
    return K,polygon,plane,points


def test_depth_plane_and_reviewed_contour_preserve_projection_and_thickness():
    K,uv,p,points=inputs()
    v,f,report=slab(K,uv,p,thickness_m=.02,observed_points=points,evidence='Reviewed visible support edges')
    projected=v[:4]@K.T
    assert np.allclose(projected[:,:2]/projected[:,2,None],uv)
    assert np.allclose(v[:4]@p[:3]+p[3],0)
    assert np.allclose(v[4:]@p[:3]+p[3],-.02)
    assert np.isclose(report['volume_m3'],.8*.8*.02)
    assert report['object_or_task_targets_used'] is False
    import trimesh
    assert trimesh.Trimesh(v,f,process=False).is_watertight


@pytest.mark.parametrize('mode',['missing_depth','wrong_plane','crossed_outline','unbounded_thickness'])
def test_unsupported_hypotheses_are_rejected(mode):
    K,uv,p,points=inputs();thickness=.02
    if mode=='missing_depth':points=points[:10]
    if mode=='wrong_plane':points[:,2]=3.
    if mode=='crossed_outline':uv=uv[[0,2,1,3]]
    if mode=='unbounded_thickness':thickness=.5
    with pytest.raises(ValueError):
        slab(K,uv,p,thickness_m=thickness,observed_points=points,evidence='Reviewed perimeter')


def test_later_visible_surface_plane_rejects_foreground_outliers():
    from real2sim2real.retargeting.planar_outline import fit_plane,mask_outline
    _,_,_,points=inputs();outliers=points[:20].copy();outliers[:,2]=1.2
    plane,report=fit_plane(np.r_[points,outliers],np.array([0.,0,-1.]))
    assert np.allclose(plane,[0,0,-1,2],atol=1e-8)
    assert report['inlier_fraction']>.8
    mask=np.zeros((100,100),bool);mask[20:80,25:75]=True
    contour=mask_outline(mask)
    assert set(map(tuple,contour))=={(25.,20.),(74.,20.),(74.,79.),(25.,79.)}


def test_heldout_depth_rejects_shift_even_when_image_outline_is_unchanged():
    from real2sim2real.retargeting.planar_outline import inspect_plane
    _,_,plane,points=inputs()
    assert inspect_plane(points,plane)['passed']
    shifted=points.copy();shifted[:,2]+=.015
    result=inspect_plane(shifted,plane)
    assert not result['passed']
    assert result['plane_refitted'] is False
    assert np.isclose(result['median_residual_m'],.015)
    assert np.allclose(plane,[0,0,-1,2])
