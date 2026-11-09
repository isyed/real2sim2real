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


def test_explicit_open_mesh_proxy_preserves_collision_mesh_and_records_assumption(tmp_path):
    import json
    import xml.etree.ElementTree as ET
    mesh=trimesh.creation.box(extents=[.07,.13,.15]);mesh.faces=mesh.faces[:-1]
    source=tmp_path/'book.obj';mesh.export(source);original=source.read_bytes()
    write_urdf('book.obj',tmp_path/'book.urdf',inertia_geometry='convex_hull_if_open')
    result=json.loads((tmp_path/'book.inertia.json').read_text())
    root=ET.parse(tmp_path/'book.urdf').getroot()
    assert result['inertia_geometry']=='convex_hull_proxy'
    assert result['source_watertight'] is False and result['inertia_proxy_approximate'] is True
    assert result['mesh_volume_m3']>0 and np.linalg.eigvalsh(result['inertia_kg_m2']).min()>0
    assert source.read_bytes()==original
    assert root.find('link/collision/geometry/mesh').get('filename')=='book.obj'


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


def test_inertial_pair_rejects_geometry_mass_and_nonphysical_changes(tmp_path):
    from real2sim2real.retargeting.object_inertia import audit_inertial_pair
    mesh=trimesh.creation.box(extents=[.2,.3,.4]);mesh.export(tmp_path/'mesh.obj')
    write_urdf('mesh.obj',tmp_path/'original.urdf')
    import xml.etree.ElementTree as ET
    source=tmp_path/'original.urdf';target=tmp_path/'candidate.urdf'
    tree=ET.parse(source);tree.find('link/inertial/origin').set('xyz','.01 0 0');tree.write(target)
    result=audit_inertial_pair(source,target)
    assert result['geometry_unchanged'] and result['mass_unchanged']
    assert not result['physical_stability_verified']
    for node,key,value in [('link/inertial/mass','value','2'),('link/visual/origin','xyz','.01 0 0'),
                           ('link/inertial/inertia','ixx','-1')]:
        tree=ET.parse(source);tree.find(node).set(key,value);tree.write(target)
        with pytest.raises(ValueError):audit_inertial_pair(source,target)


def test_inertial_pair_detects_different_mesh_bytes_at_same_relative_path(tmp_path):
    from real2sim2real.retargeting.object_inertia import audit_inertial_pair
    for name,scale in [('before',1),('after',2)]:
        folder=tmp_path/name;folder.mkdir();trimesh.creation.box(extents=[scale]*3).export(folder/'mesh.obj')
        write_urdf('mesh.obj',folder/'object.urdf')
    with pytest.raises(ValueError,match='geometry'):
        audit_inertial_pair(tmp_path/'before/object.urdf',tmp_path/'after/object.urdf')
