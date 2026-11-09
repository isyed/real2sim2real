"""Package a bounded encoder-reference experiment, retaining immutable inputs."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from real2sim2real.artifacts import digest, load, publish, write_json


def package(source, output, options):
    from flash_chord.runtime.actions.reference_denoising import ReferenceDenoisingConfig
    from .frozen import copy_tree, materialize

    source, output = Path(source).resolve(), Path(output).resolve()
    cfg = ReferenceDenoisingConfig(**options)
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a fresh independent policy directory")
    manifest, _ = materialize(source)
    action = manifest['config']['action']
    if (action.get('reference_denoising') is not None or action.get('approach_heading') is not None
            or action.get('reference_pull_gain', 0) != 0):
        raise ValueError("Do not stack experimental body controllers")
    receipt = load(source, kinds={'frozen-controller'})
    parent_sha = digest(source/'manifest.json')
    copy_tree(source, output, hardlink=True)
    action['reference_denoising'] = dict(
        _target_='flash_chord.runtime.actions.reference_denoising.ReferenceDenoisingConfig', **asdict(cfg))
    manifest['config']['env']['action'] = deepcopy(action)
    manifest['controller'] = 'frozen_sonic_reference_conditioning_fixed_hands'
    manifest['reference_conditioning_parent_manifest_sha256'] = parent_sha
    manifest['reference_conditioning_experimental'] = True
    write_json(output/'manifest.json', manifest)
    (output/'artifact.json').unlink()
    publish(output, 'frozen-controller',
            {k: output/v['path'] for k,v in receipt['files'].items()},
            metadata=dict(receipt['metadata'], physics_validated=False,
                          reference_conditioning_experimental=True, parent_policy_sha256=digest(source/'artifact.json')))
    materialize(output)
    if digest(source/'manifest.json') != parent_sha:
        raise ValueError("Parent policy mutated")
    return output
