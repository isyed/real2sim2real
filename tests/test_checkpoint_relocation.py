import json
import struct
import numpy as np
import pytest
from safetensors.numpy import save_file, load_file
from real2sim2real.controller_actor_training.relocate_checkpoint import relocate


def test_relocation_preserves_tensor_payload_and_training_config(tmp_path):
    source, output = tmp_path/'source.safetensors', tmp_path/'deployment.safetensors'
    original = json.dumps({'task': {'parquet': '/old/reference.parquet'}})
    save_file({'x': np.arange(6, dtype=np.float32)}, str(source),
              metadata={'user.resolved_config_json': original, 'user.environment_steps': '5000'})
    receipt = relocate(source, output, {'task.parquet': ('/old/reference.parquet', '/new/reference.parquet')},
                       evidence={'reference_sha256_verified': 'example'})
    def unpack(path):
        raw = path.read_bytes(); n = struct.unpack('<Q', raw[:8])[0]
        return json.loads(raw[8:8+n]), raw[8+n:]
    old, a = unpack(source); new, b = unpack(output)
    assert a == b and old['x'] == new['x']
    assert new['__metadata__']['user.original_resolved_config_json'] == original
    assert receipt['tensors_changed'] is False
    np.testing.assert_array_equal(load_file(str(source))['x'], load_file(str(output))['x'])
    with pytest.raises(ValueError):
        relocate(source, output, {'task.parquet': ('/old/reference.parquet', '/new/reference.parquet')}, evidence={'x': 1})


def test_reject_configuration_change_disguised_as_relocation(tmp_path):
    source = tmp_path/'source.safetensors'
    save_file({'x': np.ones(1)}, str(source), metadata={'user.resolved_config_json': '{"action":{"gain":"/old"}}'})
    with pytest.raises(ValueError):
        relocate(source, tmp_path/'new.safetensors', {'action.gain': ('/old', '/new')}, evidence={'x': 1})


def test_explicit_deployment_chain_preserves_original_config_and_payload(tmp_path):
    source=tmp_path/'original.safetensors';a=tmp_path/'a.safetensors';b=tmp_path/'b.safetensors'
    original=json.dumps({'task':{'parquet':'/original/reference'}})
    save_file({'x':np.arange(3,dtype=np.float32)},str(source),metadata={'user.resolved_config_json':original})
    first=relocate(source,a,{'task.parquet':('/original/reference','/a/reference')},evidence={'asset_hash':'same'})
    with pytest.raises(ValueError,match='original training'):
        relocate(a,b,{'task.parquet':('/a/reference','/b/reference')},evidence={'asset_hash':'same'})
    last=relocate(a,b,{'task.parquet':('/a/reference','/b/reference')},evidence={'asset_hash':'same'},allow_deployment_chain=True)
    assert last['parent_deployment']['tensor_payload_sha256']==first['tensor_payload_sha256']==last['tensor_payload_sha256']
    raw=b.read_bytes();size=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+size])
    assert header['__metadata__']['user.original_resolved_config_json']==original
    np.testing.assert_array_equal(load_file(str(source))['x'],load_file(str(b))['x'])
