import numpy as np
import trimesh
from real2sim2real.controller_actor_training.display_mesh import simplify


def test_truncated_result_is_rejected_and_source_retained(monkeypatch):
    source=trimesh.creation.icosphere(subdivisions=2)
    bad=trimesh.creation.box(extents=[1,1,.1])
    monkeypatch.setattr(trimesh.Trimesh,'simplify_quadric_decimation',lambda self,**kw:bad)
    mesh,report=simplify(source,20)
    assert mesh is source and report['method']=='source_fallback'
    assert not report['attempts'][0]['passed']


def test_good_bounds_allow_budgeted_proxy(monkeypatch):
    source=trimesh.creation.icosphere(subdivisions=2)
    good=trimesh.creation.box(extents=[2,2,2])
    monkeypatch.setattr(trimesh.Trimesh,'simplify_quadric_decimation',lambda self,**kw:good)
    mesh,report=simplify(source,20)
    assert mesh is good and report['method']=='guarded_decimation'
    assert report['attempts'][0]['passed']


def test_small_mesh_is_not_decimated():
    source=trimesh.creation.box();mesh,report=simplify(source,30)
    assert mesh is source and report['method']=='source_mesh'
    np.testing.assert_array_equal(mesh.vertices,source.vertices)
