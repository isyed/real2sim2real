import numpy as np
import trimesh
from PIL import Image
from real2sim2real.reconstruction.mesh_appearance import foundationpose_visual


def geometry():
    return np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]])


def test_existing_vertex_colours_are_not_sent_to_missing_to_color():
    v,f=geometry()
    colors=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]],np.uint8)
    mesh=trimesh.Trimesh(v,f,vertex_colors=colors,process=False)
    assert not hasattr(mesh.visual,"to_color")
    result=foundationpose_visual(mesh)
    assert np.array_equal(result.visual.vertex_colors,colors)
    assert np.array_equal(result.vertices,v)
    assert result.metadata["tracking_appearance"]=="preserved_vertex"


def test_texture_is_converted_to_real_colors():
    v,f=geometry()
    visual=trimesh.visual.texture.TextureVisuals(uv=np.array([[0.,0],[1,0],[0,1]]),
        image=Image.new("RGB",(4,4),(40,90,180)))
    mesh=trimesh.Trimesh(v,f,visual=visual,process=False)
    result=foundationpose_visual(mesh)
    assert np.array_equal(result.visual.vertex_colors[:,:3],np.tile([40,90,180],(3,1)))


def test_actual_missing_appearance_is_labelled():
    v,f=geometry();mesh=trimesh.Trimesh(v,f,process=False)
    result=foundationpose_visual(mesh)
    assert result.metadata["tracking_appearance"]=="missing_source_visual_default_grey"
    assert np.all(result.visual.vertex_colors[:,:3]==180)


def test_actual_export_loader_does_not_replace_vertex_colors_with_grey(tmp_path):
    # Exercise the real loader body without importing its CUDA-only module setup.
    import ast
    from pathlib import Path
    source=Path(__file__).parents[1]/"modules/reconstruction/src/r2s2r_engines/cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py"
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="load_export_object_mesh")
    namespace=dict(Path=Path,trimesh=trimesh,np=np,
        resolve_object_mesh_path=lambda p:Path(p),
        load_object_mesh_parts=lambda p:(trimesh.load(p,force="mesh",process=False),),
        concatenate_object_mesh_parts=trimesh.util.concatenate)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    v,f=geometry();colors=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]],np.uint8)
    path=tmp_path/"actual.glb";trimesh.Trimesh(v,f,vertex_colors=colors,process=False).export(path)
    result=namespace["load_export_object_mesh"](path)
    assert np.array_equal(result.visual.vertex_colors,colors)
