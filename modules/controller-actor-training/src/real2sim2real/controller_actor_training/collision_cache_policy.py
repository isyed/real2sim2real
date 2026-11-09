"""Pin existing collision hulls without rebuilding a policy's reference clock."""
from pathlib import Path
from real2sim2real.artifacts import load,digest,publish,write_json
from .frozen import materialize,copy_tree,copy_collision_cache


def package(source,cache,output,*,evidence,hardlink=False):
    source,cache,output=map(lambda p:Path(p).resolve(),[source,cache,output])
    if output.exists() or output.is_relative_to(source) or not isinstance(evidence,dict) or not evidence:
        raise ValueError('Fresh independent output and explicit cache provenance required')
    original=load(source,kinds={'frozen-controller'});manifest,_=materialize(source)
    source_hash=digest(source/'artifact.json');copy_tree(source,output,hardlink=hardlink)
    temporary=output/'validated_cache_additions'
    names=copy_collision_cache(cache,temporary,hardlink=hardlink)
    target=output/'assets/.cache/convex';target.mkdir(parents=True,exist_ok=True)
    for name in names:
        incoming=temporary/name;dest=target/name
        if dest.exists():
            if digest(dest)!=digest(incoming):raise ValueError('Same collision key has different hull data')
            incoming.unlink()
        else:incoming.rename(dest)
    temporary.rmdir()
    manifest['collision_cache_files']=sorted(set(manifest.get('collision_cache_files',[]))|set(names))
    write_json(output/'manifest.json',manifest)
    receipt=dict(schema='real2sim2real.collision-cache-policy/1',source_artifact_sha256=source_hash,
        cache_files={name:digest(target/name) for name in names},evidence=evidence,
        reference_clock_rebuilt=False,reference_geometry_or_controller_changed=False,
        original_physical_evidence_scope_preserved=True,physical_equivalence_established=False,
        cache_provenance_requires_review=True)
    write_json(output/'collision_cache_pinning.json',receipt);(output/'artifact.json').unlink()
    files={f'file_{i}':p for i,p in enumerate(sorted(output.rglob('*'))) if p.is_file()}
    publish(output,'frozen-controller',files,metadata=dict(original['metadata'],
        source_artifact_sha256=source_hash,collision_cache_pinned=True))
    materialize(output);load(source,kinds={'frozen-controller'})
    if digest(source/'artifact.json')!=source_hash:raise ValueError('Parent changed')
    return receipt
