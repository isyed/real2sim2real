"""Prepare an explicit full-length translation candidate for diagnostic handoffs.

Source quality is retained as parent evidence, never relabelled as candidate
quality. Packaging is not reconstruction, physics or actor-training approval.
"""
from pathlib import Path
import os,shutil
import numpy as np
from real2sim2real.artifacts import load,publish,digest,write_json


def prepare(source, geometry, output, *, evidence, hardlink=False):
    source,geometry,output=Path(source),Path(geometry),Path(output)
    original=load(source,kinds={'reconstruction'})
    if evidence.get('diagnostic_only') is not True or evidence.get('clean_motion_used') is not False:
        raise ValueError('Explicit source-only diagnostic evidence is required')
    if evidence.get('parent_artifact_sha256')!=digest(source/'artifact.json'):
        raise ValueError('Candidate evidence belongs to another parent')
    if evidence.get('geometry_sha256')!=digest(geometry):
        raise ValueError('Candidate geometry changed')
    with np.load(source/original['files']['geometry']['path']) as z:before=dict(z)
    with np.load(geometry) as z:after=dict(z)
    if set(before)!=set(after):raise ValueError('Candidate geometry fields differ')
    for name,value in before.items():
        if value.shape!=after[name].shape or not np.isfinite(after[name]).all():
            raise ValueError('Invalid complete candidate shape or values: '+name)
        if name!='object_translation' and not np.array_equal(value,after[name]):
            raise ValueError('Translation candidate changed immutable field: '+name)
    if len(after['frame_ids'])<2 or np.any(np.diff(after['frame_ids'])!=1):
        raise ValueError('A complete contiguous source timeline is required')
    changed=np.flatnonzero(np.any(before['object_translation']!=after['object_translation'],axis=1))
    report=dict(schema='real2sim2real.translation-candidate/1',evidence=evidence,
        source_frames=len(after['frame_ids']),changed_frame_ids=after['frame_ids'][changed].tolist(),
        complete_timeline=True,immutable_fields_preserved=True,diagnostic_only=True,
        original_quality_applies_to_parent_only=True,metric_validated=False,
        full_reference_approved=False,physics_validated=False,actor_training_approved=False)
    if output.exists():raise FileExistsError('Preserve prior candidate')
    output.mkdir(parents=True);files={}
    for role,member in original['files'].items():
        incoming=geometry if role=='geometry' else source/member['path']
        name=('parent_'+role if role in ('quality','diagnostics') else role)+incoming.suffix
        target=output/name
        if hardlink:os.link(incoming,target)
        else:shutil.copy2(incoming,target)
        files['parent_'+role if role in ('quality','diagnostics') else role]=target
    write_json(output/'candidate_report.json',report);files['candidate_report']=output/'candidate_report.json'
    publish(output,'reconstruction',files,metadata=dict(original['metadata'],
        parent_artifact_sha256=digest(source/'artifact.json'),diagnostic_only=True,
        quality_scope='Parent quality is not a remeasurement of the candidate',
        physical_execution_validated=False,actor_training_approved=False))
    return report
