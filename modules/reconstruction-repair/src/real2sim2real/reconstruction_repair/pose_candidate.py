"""Explicit complete pose candidates with interval provenance and seam checks.

An application can accept a recorded comparison warning without relabelling its
gate as passed. Composition never interpolates, rescales geometry, or certifies
physics. Source pose IDs and the decision for every replacement stay explicit.
"""
from pathlib import Path
import os, shutil
import numpy as np
from scipy.spatial.transform import Rotation
from real2sim2real.artifacts import load, publish, digest, write_json
from real2sim2real.reconstruction.recovery_quality import rigid


def compose(geometry, segments, *, max_step_m=.08, max_rotation_step_deg=30.):
    if not np.isfinite([max_step_m,max_rotation_step_deg]).all() or min(max_step_m,max_rotation_step_deg)<=0:
        raise ValueError('Positive finite seam limits required')
    result={k:np.asarray(v).copy() for k,v in geometry.items()}
    ids=result['frame_ids'];n=len(ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or n<2 or np.any(np.diff(ids)!=1):
        raise ValueError('Complete consecutive native source timeline required')
    owner=np.zeros(n,dtype=int);sources=[dict(name='complete_parent',source_frame_ids=ids.tolist())]
    for index,segment in enumerate(segments,1):
        selected=np.asarray(segment['source_frame_ids']);poses=np.asarray(segment['poses_camera'])
        evidence=segment['evidence']
        if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or not len(selected)
                or np.any(np.diff(selected)!=1) or not np.isin(selected,ids).all()
                or poses.shape!=(len(selected),4,4) or not all(rigid(p) for p in poses)
                or not evidence.get('source_sha256') or not evidence.get('name')):
            raise ValueError('Hashed named rigid source interval required')
        if evidence.get('failed_gates') and evidence.get('explicit_warning_acceptance') is not True:
            raise ValueError('Recorded failed gates require an explicit warning acceptance')
        positions=np.searchsorted(ids,selected)
        result['object_rotation'][positions]=poses[:,:3,:3]
        result['object_translation'][positions]=poses[:,:3,3]
        owner[positions]=index;sources.append(dict(evidence,source_frame_ids=selected.tolist()))
    poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,:3]=result['object_rotation'];poses[:,:3,3]=result['object_translation']
    if not all(rigid(p) for p in poses):raise ValueError('Complete rigid pose timeline required')
    step=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    angle=np.rad2deg((Rotation.from_matrix(poses[1:,:3,:3])*Rotation.from_matrix(poses[:-1,:3,:3]).inv()).magnitude())
    bad=np.flatnonzero((step>max_step_m)|(angle>max_rotation_step_deg))+1
    joins=np.flatnonzero(np.diff(owner)!=0)+1
    # Complete storage and qualified replacement coverage are different claims.
    # Expose fallback ownership and failed joins even when every source interval
    # was independently accepted before composition.
    join_rows=[dict(source_frame=int(ids[i]),step_m=float(step[i-1]),
        rotation_step_deg=float(angle[i-1]),
        passed=bool(step[i-1]<=max_step_m and angle[i-1]<=max_rotation_step_deg),
        previous_source=sources[owner[i-1]]['name'],
        next_source=sources[owner[i]]['name']) for i in joins]
    return result,dict(schema='real2sim2real.interval-pose-candidate/1',sources=sources,
        source_owner_indices=owner.tolist(),source_frame_ids=ids.tolist(),complete_timeline=True,
        parent_fallback_frames=int((owner==0).sum()),replacement_frames=int((owner!=0).sum()),
        failed_join_frame_ids=[row['source_frame'] for row in join_rows if not row['passed']],
        joins_passed=all(row['passed'] for row in join_rows),
        replacement_interpolation=False,geometry_scaled=False,
        joins=join_rows,
        failed_motion_frame_ids=ids[bad].tolist(),motion_bounds_passed=not len(bad),
        max_step_m=float(step.max()),max_rotation_step_deg=float(angle.max()),
        limits=dict(step_m=max_step_m,rotation_step_deg=max_rotation_step_deg),
        original_gate_failures_preserved=True,complete_reference_approved=False,physics_validated=False)


def prepare(source, geometry, output, *, evidence, hardlink=False, extra_files=None):
    source,geometry,output=Path(source),Path(geometry),Path(output)
    parent=load(source,kinds={'reconstruction'})
    if (evidence.get('clean_motion_used') is not False or not evidence.get('decision')
            or evidence.get('parent_artifact_sha256')!=digest(source/'artifact.json')
            or evidence.get('geometry_sha256')!=digest(geometry)):
        raise ValueError('Explicit source-only, hash-bound candidate decision required')
    with np.load(source/parent['files']['geometry']['path']) as z:before=dict(z)
    with np.load(geometry) as z:after=dict(z)
    if set(before)!=set(after):raise ValueError('Candidate geometry fields differ')
    for key,value in before.items():
        if value.shape!=after[key].shape or not np.isfinite(after[key]).all():raise ValueError('Incomplete candidate: '+key)
        if key not in ('object_translation','object_rotation') and not np.array_equal(value,after[key]):
            raise ValueError('Pose candidate changed immutable field: '+key)
    n=len(after['frame_ids']);poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,:3]=after['object_rotation'];poses[:,:3,3]=after['object_translation']
    if n<2 or np.any(np.diff(after['frame_ids'])!=1) or not all(rigid(p) for p in poses):
        raise ValueError('Complete rigid candidate timeline required')
    if output.exists():raise FileExistsError('Preserve previous candidate')
    output.mkdir(parents=True);files={}
    for role,member in parent['files'].items():
        incoming=geometry if role=='geometry' else source/member['path']
        target_role='parent_'+role if role in ('quality','diagnostics') else role
        target=output/(target_role+incoming.suffix)
        os.link(incoming,target) if hardlink else shutil.copy2(incoming,target)
        files[target_role]=target
    for role,incoming in (extra_files or {}).items():
        if role in files or not role.replace('_','').isalnum():raise ValueError('Unique portable extra-file role required')
        incoming=Path(incoming);target=output/(role+incoming.suffix)
        os.link(incoming,target) if hardlink else shutil.copy2(incoming,target)
        files[role]=target
    report=dict(schema='real2sim2real.pose-candidate/1',evidence=evidence,complete_timeline=True,source_frames=n,
        human_mesh_camera_timing_unchanged=True,parent_quality_is_candidate_quality=False,
        full_reference_approved=False,physics_validated=False,actor_training_approved=False)
    write_json(output/'candidate_report.json',report);files['candidate_report']=output/'candidate_report.json'
    publish(output,'reconstruction',files,metadata=dict(parent['metadata'],parent_artifact_sha256=digest(source/'artifact.json'),
        candidate_decision=evidence['decision'],quality_scope='Original parent quality retained separately; see candidate report',
        physical_execution_validated=False,actor_training_approved=False))
    return report
