"""Package an explicit solver-contact experiment without editing its source."""
from copy import deepcopy
from pathlib import Path
from real2sim2real.artifacts import digest,load,publish,write_json
from .frozen import copy_tree,materialize


def package(source, output, *, enable_multiccd, evidence, hardlink=False):
    """Change only multi-contact generation; geometry and controller stay exact.

    This creates an experimental frozen policy, not a default solver change or
    a physics approval. A new backend setting requires new physical evaluation.
    """
    source,output=map(lambda p:Path(p).resolve(),[source,output])
    if (type(enable_multiccd) is not bool or not isinstance(evidence,dict) or not evidence
            or output.exists() or output.is_relative_to(source)):
        raise ValueError('Explicit boolean, evidence and fresh independent output required')
    manifest,_=materialize(source);source_hash=digest(source/'artifact.json')
    before=deepcopy(manifest['config']);config=manifest['config']
    for simulation in [config['env']['sim']]+([config['sim']] if 'sim' in config else []):
        simulation['enable_multiccd']=enable_multiccd
    receipt=dict(schema='real2sim2real.solver-contact-ablation/1',
        source_artifact_sha256=source_hash,enable_multiccd=enable_multiccd,
        previous_enable_multiccd=before['env']['sim'].get('enable_multiccd',False),
        evidence=evidence,geometry_changed=False,reference_motion_changed=False,
        controller_changed=False,training_performed=False,physical_validation=False)
    copy_tree(source,output,hardlink=hardlink)
    write_json(output/'manifest.json',manifest)
    write_json(output/'solver_contact_ablation.json',receipt)
    (output/'artifact.json').unlink()
    files={f'file_{i}':p for i,p in enumerate(sorted(output.rglob('*'))) if p.is_file()}
    publish(output,'frozen-controller',files,metadata=dict(robot='g1',actor_present=False,
        training_steps=0,physics_validated=False,solver_contact_ablation=True,
        source_artifact_sha256=source_hash))
    materialize(output)
    load(source,kinds={'frozen-controller'})
    if digest(source/'artifact.json')!=source_hash:raise ValueError('Source receipt changed')
    return receipt
