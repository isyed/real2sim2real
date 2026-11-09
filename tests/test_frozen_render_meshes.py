import numpy as np
import pytest
import trimesh
from real2sim2real.controller_actor_training.render_frozen import visual_meshes
from real2sim2real.controller_actor_training.render_frozen import support_meshes


def test_cpu_visuals_apply_mesh_scale_and_urdf_origin(tmp_path):
    mesh=trimesh.creation.box(extents=[1,2,3]);mesh.export(tmp_path/'body.obj')
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="r"><link name="body"><visual><origin xyz="1 2 3" rpy="0 0 0"/><geometry><mesh filename="body.obj" scale="2 1 1"/></geometry></visual></link></robot>')
    parts=visual_meshes(urdf,['r/body'],'r/')
    assert parts[0][0]==0
    points=parts[0][1].reshape(-1,3)
    np.testing.assert_allclose(points.min(0),[0,1,1.5])
    np.testing.assert_allclose(points.max(0),[2,3,4.5])


def test_cpu_visuals_do_not_guess_missing_body_identity(tmp_path):
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="r"><link name="body"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>')
    with pytest.raises(ValueError,match='missing'):
        visual_meshes(urdf,[],'r/')


def _support_stage(path):
    pytest.importorskip("pxr", reason="Requires the optional USD runtime")
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    parent=UsdGeom.Xform.Define(stage,'/World')
    parent.AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    mesh=UsdGeom.Mesh.Define(stage,'/World/Table')
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    stage.GetRootLayer().Save();return stage


def test_support_display_uses_simulator_triangles_and_parent_transform(tmp_path):
    path=tmp_path/'support.usda';_support_stage(path)
    parts=support_meshes(path,['g1/pelvis','support'])
    assert parts[0][0]==1
    np.testing.assert_allclose(parts[0][1][0],[[1,2,3],[2,2,3],[1,3,3]])


def test_support_display_rejects_missing_body_and_external_layers(tmp_path):
    path=tmp_path/'support.usda';stage=_support_stage(path)
    with pytest.raises(ValueError,match='body is missing'):support_meshes(path,[])
    layer=tmp_path/'other.usda';layer.write_text('#usda 1.0\n')
    stage.GetRootLayer().subLayerPaths.append(str(layer));stage.GetRootLayer().Save()
    with pytest.raises(ValueError,match='self-contained'):support_meshes(path,['support'])
