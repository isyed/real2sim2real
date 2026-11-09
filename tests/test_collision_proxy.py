import hashlib
import xml.etree.ElementTree as ET
import pytest
import trimesh
from real2sim2real.artifacts import publish,load
from real2sim2real.controller_actor_training.collision_proxy import package


def test_collision_proxy_preserves_inertia_visual_motion_and_immutable_source(tmp_path):
    source=tmp_path/'source';source.mkdir();mesh=trimesh.creation.box().subdivide().subdivide()
    mesh.export(source/'visual.obj')
    (source/'object.urdf').write_text('<robot name="box"><link name="object"><inertial><mass value="1"/><inertia ixx=".1" iyy=".2" izz=".3" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><mesh filename="visual.obj"/></geometry></visual><collision><geometry><mesh filename="visual.obj"/></geometry></collision></link></robot>')
    (source/'reference.parquet').write_bytes(b'opaque reference bytes unchanged by this asset-only tool')
    publish(source,'robot-reference',{name:source/name for name in ['object.urdf','visual.obj','reference.parquet']},metadata={'frames':3})
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    out=tmp_path/'proxy';result=package(source,out,urdf_role='object.urdf',hardlink=True)
    assert result['proxy_faces']<result['source_faces'] and not result['inertial_parameters_changed']
    assert not result['visual_geometry_changed'] and not result['physics_validated']
    assert (out/'reference.parquet').read_bytes()==(source/'reference.parquet').read_bytes()
    assert ET.parse(out/'object.urdf').find('.//visual/geometry/mesh').get('filename')=='visual.obj'
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    load(out,kinds={'robot-reference'})


def test_existing_output_cannot_overwrite_source(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'reference.parquet').write_bytes(b'reference')
    publish(source,'robot-reference',{'reference':source/'reference.parquet'},metadata={})
    with pytest.raises(ValueError):package(source,source,urdf_role='unused')
