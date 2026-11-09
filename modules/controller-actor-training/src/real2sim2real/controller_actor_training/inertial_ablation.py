"""Create an audited inertial-only policy variant without changing its source."""
from pathlib import Path
import shutil
from real2sim2real.artifacts import load,publish,digest,write_json
from .frozen import copy_tree,materialize


def package(source, candidate_urdf, output, *, evidence, hardlink=False):
    from real2sim2real.retargeting.object_inertia import audit_inertial_pair
    source,candidate_urdf,output=map(lambda p:Path(p).resolve(),[source,candidate_urdf,output])
    if output.exists() or output.is_relative_to(source) or not evidence:
        raise ValueError('Fresh independent output and explicit dynamics evidence required')
    manifest,_=materialize(source)
    urdfs=list((source/'reference').rglob('*.urdf'))
    if len(urdfs)!=1:raise ValueError('Require exactly one rigid object model')
    audit=audit_inertial_pair(urdfs[0],candidate_urdf)
    reference=load(source/'reference',kinds={'robot-reference'})
    source_hash=digest(source/'artifact.json')
    copy_tree(source,output,hardlink=hardlink)
    target=output/urdfs[0].relative_to(source);target.unlink();shutil.copy2(candidate_urdf,target)
    receipt=dict(audit=audit,evidence=evidence,source_artifact_sha256=source_hash,
                 dynamics_changed=True,training_performed=False,physical_validation=False)
    write_json(output/'reference/inertia_assumption.json',receipt)
    (output/'reference/artifact.json').unlink()
    files={role:output/'reference'/item['path'] for role,item in reference['files'].items()}
    files['inertia_assumption']=output/'reference/inertia_assumption.json'
    publish(output/'reference','robot-reference',files,metadata=dict(reference['metadata'],inertial_ablation=True))
    manifest['reference_artifact_sha256']=digest(output/'reference/artifact.json')
    write_json(output/'manifest.json',manifest)
    (output/'artifact.json').unlink()
    files={f'file_{i}':p for i,p in enumerate(sorted(output.rglob('*'))) if p.is_file()}
    publish(output,'frozen-controller',files,metadata=dict(robot='g1',actor_present=False,training_steps=0,
        physics_validated=False,reference_artifact_sha256=manifest['reference_artifact_sha256'],inertial_ablation=True))
    materialize(output)
    if digest(source/'artifact.json')!=source_hash:raise ValueError('Source receipt changed')
    load(source,kinds={'frozen-controller'})
    return receipt
