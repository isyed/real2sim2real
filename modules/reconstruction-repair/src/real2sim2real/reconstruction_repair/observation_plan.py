"""Route missing observations to input review before optimizing pose.

This is advisory triage, not evidence that a missing mask is a segmentation
error: source RGB must distinguish occlusion from a failed segmentation track.
No pose, mask, depth, acceptance threshold or reference is changed here.
"""
from collections import Counter


def plan(audit, *, semantic_review=None):
    rows=audit['rows'];n=audit['source_frames']
    if type(n) is not int or n<1 or len(rows)!=n or len({r['frame_id'] for r in rows})!=n:
        raise ValueError('Complete distinct source-frame observation audit required')
    counts=Counter(reason for row in rows for reason in set(row['reasons']))
    observed=sum(bool(r['observation_available']) for r in rows)
    if semantic_review is not None:
        identity=audit['source_identity']
        for key in ('artifact_sha256','video_sha256'):
            if not identity.get(key) or semantic_review[key]!=identity[key]:raise ValueError('Semantic review belongs to another source')
        ids=semantic_review['reviewed_frame_ids']
        if (type(semantic_review['passed']) is not bool or not semantic_review.get('evidence') or not ids
            or any(type(i) is not int or i not in {r['frame_id'] for r in rows} for i in ids)):
            raise ValueError('Explicit RGB semantic review and source frames required')
    semantic_failed=semantic_review is not None and not semantic_review['passed']
    if semantic_failed:
        action='repair_foreground_segmentation_before_pose'
        advice='Source RGB review rejected the mask target identity. Repair foreground segmentation and reassess dependent mesh/depth/pose inputs before pose optimization; good mask-overlap scores cannot validate the wrong object.'
    elif counts['object_mask_missing']>n/2:
        action='review_source_rgb_and_object_masks'
        advice='Most source frames lack an object mask. Distinguish segmentation loss from true occlusion in RGB; repair the segmentation track if visible objects were missed before another full-clip pose fit.'
    elif observed<n/2:
        action='review_foreground_visibility_and_depth'
        advice='Most frames lack reliable visible depth. Review occlusion, foreground depth and alignment before full-clip pose optimization; do not invent observations.'
    else:
        action='inspect_reported_gaps_before_pose_repair'
        advice='Review reported observation gaps and calibration separately; observation availability alone does not approve geometry or tracking.'
    return dict(schema='real2sim2real.observation-repair-plan/1',source_frames=n,
        observed_frames=observed,reason_counts=dict(counts),next_action=action,
        recommendation=advice,source_rgb_review_required=True,reference_approved=False,
        semantic_review=semantic_review,pose_optimization_blocked_by_semantic_review=semantic_failed,
        acceptance_thresholds_changed=False,hidden_motion_inferred=False)
