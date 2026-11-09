import json
import pytest
from real2sim2real.artifacts import publish,digest,load,write_json
from real2sim2real.controller_actor_training.frozen import configuration
from real2sim2real.controller_actor_training.solver_ablation import package


def fixture(tmp_path):
    source=tmp_path/'source';ref=source/'reference';ref.mkdir(parents=True)
    (ref/'reference.parquet').write_bytes(b'Immutable reference payload; no simulated execution')
    publish(ref,'robot-reference',{'reference':ref/'reference.parquet'},metadata={'robot':'g1'})
    manifest=dict(schema='real2sim2real.frozen-controller/1',actor_present=False,training_steps=0,
        correction='deterministic_zeros',reference='reference/reference.parquet',config=configuration(),
        reference_artifact_sha256=digest(ref/'artifact.json'))
    write_json(source/'manifest.json',manifest)
    publish(source,'frozen-controller',{str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()})
    return source


def test_solver_variant_preserves_source_and_reference_with_hardlinks(tmp_path):
    source=fixture(tmp_path);before=digest(source/'artifact.json');original=json.loads((source/'manifest.json').read_text())
    out=tmp_path/'variant';receipt=package(source,out,enable_multiccd=True,evidence={'control':'analytic flat object'},hardlink=True)
    load(out);load(source)
    assert digest(source/'artifact.json')==before
    assert json.loads((source/'manifest.json').read_text())==original
    result=json.loads((out/'manifest.json').read_text())
    for cfg in [original['config']['env']['sim'],original['config']['sim']]:cfg['enable_multiccd']=True
    assert result==original
    assert digest(out/'reference/artifact.json')==digest(source/'reference/artifact.json')
    assert receipt['geometry_changed'] is False and receipt['physical_validation'] is False


@pytest.mark.parametrize('value',[1,'true',None])
def test_requires_explicit_boolean_before_writing(tmp_path,value):
    source=fixture(tmp_path)
    with pytest.raises(ValueError):package(source,tmp_path/'out',enable_multiccd=value,evidence={'review':'test'})
    assert not (tmp_path/'out').exists()
