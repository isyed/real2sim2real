"""Relocate explicit configuration paths without changing checkpoint tensors.

This produces a deployment copy, not a new training checkpoint. Asset identity
and any deliberate dynamics changes must be documented by the caller.
"""
import hashlib
import json
from pathlib import Path
import struct


def relocate(source, output, replacements, *, evidence, allow_deployment_chain=False):
    """Replace only declared path fields, retaining original metadata and payload.

    replacements maps dotted configuration fields to (expected_old, new) strings.
    Unknown, non-string, relative or unexpectedly changed fields are rejected.
    """
    source, output = Path(source), Path(output)
    if output.exists() or not replacements or not evidence:
        raise ValueError('Require a fresh destination, explicit paths and evidence')
    raw = source.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    metadata = header['__metadata__']
    parent=None
    if 'user.deployment_relocation_json' in metadata:
        if not allow_deployment_chain:
            raise ValueError('Relocate the original training checkpoint, not a deployment copy')
        parent=json.loads(metadata['user.deployment_relocation_json'])
        if (parent.get('tensors_changed') is not False
                or parent.get('tensor_payload_sha256')!=hashlib.sha256(payload).hexdigest()
                or 'user.original_resolved_config_json' not in metadata):
            raise ValueError('Deployment ancestry or tensor payload is inconsistent')
    original = metadata['user.resolved_config_json']
    config = json.loads(original)
    allowed = {'task.parquet', 'reference.parquet', 'action.policy_dir', 'env.action.policy_dir'}
    for field, (old, new) in replacements.items():
        if field not in allowed or not all(isinstance(x, str) and Path(x).is_absolute() for x in (old, new)):
            raise ValueError('Only explicit absolute reference/model paths may be relocated')
        node = config
        parts = field.split('.')
        for part in parts[:-1]:
            node = node[part]
        if node[parts[-1]] != old:
            raise ValueError('Unexpected original configuration path: ' + field)
        node[parts[-1]] = new
    receipt = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        tensor_payload_sha256=hashlib.sha256(payload).hexdigest(),
        replacements=replacements, evidence=evidence, tensors_changed=False,
        training_performed=False, deployment_copy=True)
    if parent is not None:receipt['parent_deployment']=parent
    else:metadata['user.original_resolved_config_json'] = original
    metadata['user.resolved_config_json'] = json.dumps(config, separators=(',', ':'))
    metadata['user.deployment_relocation_json'] = json.dumps(receipt, separators=(',', ':'))
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        stream.write(payload)
    receipt['deployment_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    return receipt
