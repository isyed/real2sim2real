import numpy as np
import pytest
import trimesh
from real2sim2real.reconstruction.tracking_inputs import tracking_mesh


def test_rebuilt_tracking_mesh_retains_rgb_cues_and_scales_once(tmp_path):
    v=np.array([[0.,0,0],[.1,0,0],[0,.1,0]])
    f=np.array([[0,1,2]])
    color=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]],np.uint8)
    mesh=trimesh.Trimesh(v,f,vertex_colors=color,process=False)
    path=tmp_path/'colored.glb'
    mesh.export(path)
    result,receipt=tracking_mesh(dict(vertices=v,faces=f,object_scale=1.),source_mesh=path,
                                  scale=.8,require_appearance=True)
    assert np.allclose(result.vertices,v*.8)
    assert np.array_equal(result.visual.vertex_colors,color)
    assert receipt['appearance_preserved']


def test_appearance_cannot_be_attached_to_a_different_mesh_frame(tmp_path):
    v=np.array([[0.,0,0],[.1,0,0],[0,.1,0]])
    f=np.array([[0,1,2]])
    mesh=trimesh.Trimesh(v+.03,f,vertex_colors=np.full((3,4),255,np.uint8),process=False)
    path=tmp_path/'shifted.glb'
    mesh.export(path)
    data=dict(vertices=v,faces=f,object_scale=1.)
    with pytest.raises(ValueError,match='frame_or_topology'):
        tracking_mesh(data,source_mesh=path,require_appearance=True)
    _,receipt=tracking_mesh(data,source_mesh=path)
    assert not receipt['appearance_preserved']


def test_missing_appearance_is_explicit_not_an_rgb_ready_mesh():
    data=dict(vertices=np.array([[0.,0,0],[.1,0,0],[0,.1,0]]),faces=np.array([[0,1,2]]),object_scale=1.)
    _,receipt=tracking_mesh(data)
    assert not receipt['appearance_preserved']
    with pytest.raises(ValueError,match='requires verified'):
        tracking_mesh(data,require_appearance=True)


def test_texture_coordinates_survive_a_rebuild(tmp_path):
    from PIL import Image
    v=np.array([[0.,0,0],[.1,0,0],[0,.1,0]])
    f=np.array([[0,1,2]])
    uv=np.array([[0.,0.],[1.,0.],[0.,1.]])
    visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=Image.new('RGB',(4,4),(200,40,30)))
    mesh=trimesh.Trimesh(v,f,visual=visual,process=False)
    path=tmp_path/'textured.glb'
    mesh.export(path)
    result,receipt=tracking_mesh(dict(vertices=v,faces=f,object_scale=1.),source_mesh=path,
                                  require_appearance=True)
    assert receipt['appearance_preserved'] and result.visual.kind=='texture'
    assert np.allclose(result.visual.uv,uv)
