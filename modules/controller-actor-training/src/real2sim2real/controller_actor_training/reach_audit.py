"""Frame-correct palm-to-grasp-target diagnostics from saved physics states.

The reference palm's object-local position is carried by the current physical
object pose. No pose is modified and proximity is never labelled grasp success.
"""
def summarize(physical,reference,*,palm_body_ids,object_body_id,pelvis_body_id,contact_active,valid_frames=None):
    import numpy as np
    from scipy.spatial.transform import Rotation
    p,k=np.asarray(physical,float),np.asarray(reference,float)
    if p.shape!=k.shape or p.ndim!=3 or p.shape[-1]!=7 or not len(p) or not np.isfinite(p).all() or not np.isfinite(k).all():
        raise ValueError('Matching complete finite [frame, body, xyz+xyzw] states required')
    palms=tuple(palm_body_ids);ids=(*palms,object_body_id,pelvis_body_id)
    if (not palms or any(type(i) is not int or not 0<=i<p.shape[1] for i in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('Distinct explicit body IDs within the recorded layout required')
    phase=np.asarray(contact_active)
    if phase.dtype!=bool or phase.shape!=(len(p),len(palms)):
        raise ValueError('Explicit boolean [frame, palm] contact phases required')
    valid=np.ones(len(p),bool) if valid_frames is None else np.asarray(valid_frames)
    if valid.dtype!=bool or valid.shape!=(len(p),):raise ValueError('Boolean source-frame validity required')
    for value in (p,k):
        if not np.allclose(np.linalg.norm(value[:,:,3:],axis=-1),1,atol=1e-4):raise ValueError('Unit xyzw body orientations required')
    obj=object_body_id;Rp=Rotation.from_quat(p[:,obj,3:]).as_matrix();Rk=Rotation.from_quat(k[:,obj,3:]).as_matrix()
    reports=[];all_distances=[]
    for slot,body in enumerate(palms):
        relative=np.einsum('tji,tj->ti',Rk,k[:,body,:3]-k[:,obj,:3])
        wanted=np.einsum('tij,tj->ti',Rp,relative)+p[:,obj,:3]
        distance=np.linalg.norm(p[:,body,:3]-wanted,axis=1);mask=phase[:,slot]&valid
        values=distance[mask];all_distances.extend(values.tolist())
        reports.append(dict(body_id=body,measured_phase_frames=int(mask.sum()),
            median_target_error_m=float(np.median(values)) if len(values) else None,
            min_target_error_m=float(values.min()) if len(values) else None,
            frames_within_20cm=int(np.count_nonzero(values<=.2))))
    root=np.linalg.norm(p[:,pelvis_body_id,:3]-k[:,pelvis_body_id,:3],axis=1)
    return dict(schema='real2sim2real.saved-reach-audit/1',source_frames=len(p),palms=reports,
        median_target_error_m=float(np.median(all_distances)) if all_distances else None,
        mean_pelvis_tracking_error_m=float(root[valid].mean()) if valid.any() else None,
        target_frame='Reference palm carried from reference-object coordinates to current physical object pose',
        geometry_or_states_modified=False,grasp_success_established=False,
        limitation='Proximity counts are diagnostics; measured contact, stability and task completion remain separate.')
