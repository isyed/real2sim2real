from pathlib import Path
import numpy as np
import pytest
import trimesh
from real2sim2real.controller_actor_training.render_frozen import visual_meshes


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
