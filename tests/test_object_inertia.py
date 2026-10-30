import numpy as np
import pytest
import trimesh
from real2sim2real.retargeting.object_inertia import mass_properties,write_urdf


def test_translated_box_has_geometric_com_and_correct_tensor():
    mesh=trimesh.creation.box(extents=[2,4,6]);mesh.apply_translation([3,-2,1])
    result=mass_properties(mesh.vertices,mesh.faces,mass_kg=12)
    assert np.allclose(result['center_of_mass_m'],[3,-2,1])
    assert np.allclose(result['inertia_kg_m2'],np.diag([52,40,20]))


def test_open_mesh_and_invalid_mass_do_not_receive_placeholder_inertials():
    mesh=trimesh.creation.box()
    with pytest.raises(ValueError,match='closed'):mass_properties(mesh.vertices,mesh.faces[:-1])
    for value in [0,-1,float('nan'),True]:
        with pytest.raises(ValueError):mass_properties(mesh.vertices,mesh.faces,mass_kg=value)


def test_written_urdf_keeps_mesh_and_frame_but_replaces_placeholder_com(tmp_path):
    import xml.etree.ElementTree as ET
    mesh=trimesh.creation.box();mesh.apply_translation([.2,.1,.3]);source=tmp_path/'mesh.obj';mesh.export(source)
    before=source.read_bytes();write_urdf('mesh.obj',tmp_path/'object.urdf')
    root=ET.parse(tmp_path/'object.urdf').getroot()
    assert np.allclose(np.fromstring(root.find('link/inertial/origin').get('xyz'),sep=' '),[.2,.1,.3])
    assert source.read_bytes()==before
    assert root.find('link/collision/geometry/mesh').get('filename')=='mesh.obj'


def test_rotation_preserves_inertia_tensor_in_the_mesh_link_frame():
    from scipy.spatial.transform import Rotation
    mesh=trimesh.creation.box(extents=[.2,.4,.8])
    original=np.asarray(mass_properties(mesh.vertices,mesh.faces)['inertia_kg_m2'])
    R=Rotation.from_euler('xyz',[.3,.4,.7]).as_matrix()
    rotated=mass_properties(mesh.vertices@R.T,mesh.faces)
    assert np.allclose(rotated['inertia_kg_m2'],R@original@R.T)
