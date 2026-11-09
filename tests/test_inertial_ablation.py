import json
import pytest
import trimesh
from real2sim2real.artifacts import publish,load,write_json,digest
from real2sim2real.controller_actor_training.frozen import configuration
from real2sim2real.controller_actor_training.inertial_ablation import package
from real2sim2real.retargeting.object_inertia import write_urdf


def fixture(tmp_path):
    source=tmp_path/'source';ref=source/'reference';ref.mkdir(parents=True)
    trimesh.creation.box(extents=[.2,.3,.4]).export(ref/'mesh.obj')
    write_urdf('mesh.obj',ref/'object.urdf')
    (ref/'reference.parquet').write_bytes(b'opaque reference payload; this test does not run a simulator')
    publish(ref,'robot-reference',{p.name:p for p in ref.iterdir()},metadata={'robot':'g1'})
    manifest=dict(schema='real2sim2real.frozen-controller/1',actor_present=False,training_steps=0,
        correction='deterministic_zeros',reference='reference/reference.parquet',config=configuration(),
        reference_artifact_sha256=digest(ref/'artifact.json'))
    write_json(source/'manifest.json',manifest)
    publish(source,'frozen-controller',{str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()})
    candidate=tmp_path/'candidate';candidate.mkdir()
    (candidate/'mesh.obj').write_bytes((ref/'mesh.obj').read_bytes())
    import xml.etree.ElementTree as ET
    tree=ET.parse(ref/'object.urdf');tree.find('link/inertial/origin').set('xyz','.01 0 0');tree.write(candidate/'object.urdf')
    return source,candidate/'object.urdf'


def test_hardlinked_variant_keeps_source_and_updates_nested_hashes(tmp_path):
    source,candidate=fixture(tmp_path);old=load(source);original=digest(source/'reference/object.urdf')
    out=tmp_path/'out';receipt=package(source,candidate,out,evidence={'assumption':'test'},hardlink=True)
    assert load(source)==old and digest(source/'reference/object.urdf')==original
    load(out);load(out/'reference')
    assert digest(out/'reference/object.urdf')!=original
    assert digest(out/'reference/reference.parquet')==digest(source/'reference/reference.parquet')
    assert json.loads((out/'manifest.json').read_text())['config']==json.loads((source/'manifest.json').read_text())['config']
    assert receipt['dynamics_changed'] and not receipt['physical_validation']


def test_invalid_geometry_is_rejected_before_creating_variant(tmp_path):
    source,candidate=fixture(tmp_path)
    (candidate.parent/'mesh.obj').write_bytes(b'changed mesh')
    with pytest.raises(ValueError,match='geometry'):
        package(source,candidate,tmp_path/'out',evidence={'assumption':'test'})
    assert not (tmp_path/'out').exists()
