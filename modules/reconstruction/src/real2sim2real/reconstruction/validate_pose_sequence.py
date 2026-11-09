"""Check proposed poses with the actual GPU renderer and native evidence.

No pose fitting, tracking carry-forward or physical execution happens here.
Rejected frames stop a measured prefix; a renderer pass is not metric truth.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .recovery_quality import rigid,consistency,sample_vertices,rejection_reasons


def local_indices(source_ids,window_ids):
    ids,window=np.asarray(source_ids),np.asarray(window_ids)
    if (ids.ndim!=1 or window.ndim!=1 or not len(ids) or not len(window)
        or ids.dtype.kind not in 'iu' or window.dtype.kind not in 'iu'
        or np.any(ids<0) or np.any(window<0) or np.any(np.diff(ids)!=1) or np.any(np.diff(window)!=1)
        or not np.isin(ids,window).all()):raise ValueError('Consecutive explicit native source IDs required')
    return np.searchsorted(window,ids)


def sequence_boundary(proposals, source_ids):
    """Only native frame zero may lack history; do not invent a prior frame."""
    poses=np.asarray(proposals['poses_camera'],np.float32)
    ids=np.asarray(source_ids)
    if poses.shape!=(len(ids),4,4) or not all(rigid(p) for p in poses):
        raise ValueError('Rigid proposed poses required')
    has_pose='previous_pose_camera' in proposals
    has_id='previous_source_frame' in proposals
    if has_pose!=has_id:raise ValueError('Previous pose and source ID must be paired')
    if not has_pose:
        if ids[0]!=0:raise ValueError('Only native frame zero may omit previous motion history')
        return poses,poses[0].copy(),True
    previous=np.asarray(proposals['previous_pose_camera'],np.float32)
    preceding=np.asarray(proposals['previous_source_frame'])
    if (not rigid(previous) or preceding.shape!=() or preceding.dtype.kind not in 'iu'
            or int(preceding)!=ids[0]-1 or int(preceding)<0):
        raise ValueError('Explicit rigid preceding native pose required')
    return poses,previous,False


def source_support(support, source_frame):
    """Select explicit native-frame support; never carry another frame's plane."""
    if support.get('ground_truth_used') is not False or not support.get('evidence'):
        raise ValueError('Explicit predicted support required')
    if 'frames' not in support:
        return support
    rows=support['frames']
    if not isinstance(rows,list) or not rows:
        raise ValueError('Nonempty native support records required')
    ids=[r.get('source_frame') for r in rows]
    if any(type(i) is not int or i<0 for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('Unique increasing native support IDs required')
    matches=[r for r in rows if r['source_frame']==source_frame]
    if len(matches)!=1:raise ValueError('Missing native support for source frame')
    row=matches[0]
    if not all(k in row for k in ('plane_camera','center_camera_m','radius_m')):
        raise ValueError('Complete native support geometry required')
    return row


def validate(export,depth,support,proposals,mapping,output,*,tracking_evidence=None):
    from .tracking_recovery import load_exporter
    from real2sim2real.artifacts import digest,write_json
    export,depth,support,proposals,mapping,output=map(Path,[export,depth,support,proposals,mapping,output])
    if output.exists():raise FileExistsError('Preserve prior sequence validation')
    if (export/'poses.npy').exists():raise ValueError('Prediction-only export required')
    source=json.loads(mapping.read_text());info=json.loads((export/'wild_export.json').read_text());plane=json.loads(support.read_text())
    if source['source_video_sha256']!=info['source_video_sha256']:raise ValueError('Source identities differ')
    if plane.get('ground_truth_used') is not False or not plane.get('evidence'):raise ValueError('Explicit predicted support required')
    with np.load(proposals) as z:
        ids=np.asarray(z['source_frame_ids']);indices=local_indices(ids,source['source_frame_indices'])
        poses,previous,initial_without_history=sequence_boundary(z,ids)
    supports=[source_support(plane,int(i)) for i in ids]
    recoveries=np.ones(len(ids),dtype=bool)
    policy=dict(validation_policy='all_recovery_strength',shared_thresholds_changed=False)
    if tracking_evidence is not None:
        from .tracked_pose_evidence import recovery_flags
        recoveries,policy=recovery_flags(ids,poses,tracking_evidence,source_video_sha256=source['source_video_sha256'])
        policy['validation_policy']='provenance_matched_tracking_or_recovery'
    output.mkdir(parents=True);(output/'debug').mkdir();up=load_exporter()
    from prep.mhr_export_utils import frame_names
    # Validate the real RGB-cache contract before constructing GPU models.
    # An existing HDF5 file is not evidence of a readable native timeline.
    names=frame_names(export,0)
    K=np.asarray(info['intrinsics'])
    for local in sorted(set([int(indices[0]),int(indices[-1])])):
        up._read_foundationpose_frame(export,depth,0,names[local],K)
    mesh=up.load_export_object_mesh(export);est=up._pose_estimator(mesh,output/'debug')
    transform=up._tensor_to_numpy(est.get_tf_to_centered_mesh());points=sample_vertices(est.mesh_ori.vertices);rows=[]
    for native,local,pose,native_support,recovery in zip(ids,indices,poses,supports,recoveries):
        _,_,z,human,obj,k=up._read_foundationpose_frame(export,depth,0,names[int(local)],K)
        mask=up._render_candidate_masks(est,(pose@np.linalg.inv(transform))[None],k,obj.shape)
        metrics=up._selected_metric_values(up.measure_visible_silhouettes(mask,human,obj),0)
        geometry=consistency(points,pose,k,z,obj&~human,native_support)
        rejected=rejection_reasons(pose,previous,metrics,geometry,recovery=bool(recovery))
        row=dict(source_frame=int(native),tracker_local_frame=int(local),metrics=metrics,geometry=geometry,rejected_for=rejected,accepted=not rejected,validation_mode='recovery' if recovery else 'tracking',
                 motion_history_checked=not (initial_without_history and int(native)==0))
        rows.append(row);write_json(output/'progress.json',dict(checked=len(rows),total=len(ids),last=row))
        if rejected:break
        previous=pose
    result=dict(schema='real2sim2real.native-pose-sequence-validation/1',rows=rows,checked_frames=len(rows),total_frames=len(ids),
        accepted_frames=sum(r['accepted'] for r in rows),complete=len(rows)==len(ids) and all(r['accepted'] for r in rows),
        proposal_sha256=digest(proposals),depth_sha256=digest(depth),mapping_sha256=digest(mapping),support_sha256=digest(support),
        thresholds_relaxed=False,poses_changed=False,physics_validated=False,complete_reference_approved=False,validation_policy=policy,
        initial_frame_without_motion_history=initial_without_history)
    write_json(output/'RESULTS.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['export','depth','support','proposals','mapping','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--tracking-evidence',help='Optional hash-bound accepted tracking evidence. Default remains strict recovery validation for every pose.')
    a=p.parse_args();validate(a.export,a.depth,a.support,a.proposals,a.mapping,a.output,tracking_evidence=a.tracking_evidence)
