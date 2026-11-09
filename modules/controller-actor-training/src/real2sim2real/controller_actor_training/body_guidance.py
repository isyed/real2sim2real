"""Derive an auditable opt-in body-guidance policy without changing its source."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy

from real2sim2real.artifacts import digest, load, publish, write_json


def package(source, output, guidance):
    from flash_chord.runtime.actions.approach_heading import ApproachHeadingConfig
    from .frozen import materialize, copy_tree
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Use a fresh independent policy directory')
    guidance = ApproachHeadingConfig(**guidance)
    manifest, _ = materialize(source)
    if manifest['config']['action'].get('approach_heading') is not None:
        raise ValueError('Do not stack body-guidance revisions')
    if manifest['config']['action'].get('reference_pull_gain', 0) != 0:
        raise ValueError('Existing reference pull conflicts with early leg guidance')
    receipt = load(source, kinds={'frozen-controller'})
    copy_tree(source, output, hardlink=True)
    # Atomic JSON replacement leaves hardlinked source bytes unchanged.
    action = manifest['config']['action']
    action['approach_heading'] = dict(
        _target_='flash_chord.runtime.actions.approach_heading.ApproachHeadingConfig', **asdict(guidance))
    manifest['config']['env']['action'] = deepcopy(action)
    manifest['controller'] = 'frozen_sonic_early_body_guidance_fixed_hands'
    manifest['body_guidance_parent_manifest_sha256'] = digest(source/'manifest.json')
    manifest['body_guidance_experimental'] = True
    write_json(output/'manifest.json', manifest)
    (output/'artifact.json').unlink()
    files = {role: output/row['path'] for role, row in receipt['files'].items()}
    publish(output, 'frozen-controller', files, metadata=dict(receipt['metadata'],
        physics_validated=False, body_guidance_experimental=True,
        parent_policy_sha256=digest(source/'artifact.json')))
    materialize(output)
    return output


def summarize(body_q, reference_q, *, fps, pelvis=0, palms=(30,45), early_seconds=10.):
    """Root/heading and palm errors on explicit aligned control frames.

    Callers must declare any post-step vs pre-step sample offset separately.
    Low pelvis is a diagnostic only, not a complete balance/success classifier.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    q,r=np.asarray(body_q),np.asarray(reference_q)
    if (q.ndim!=3 or q.shape!=r.shape or q.shape[-1]!=7 or len(q)<2
            or not np.isfinite(q).all() or not np.isfinite(r).all()
            or not np.isfinite(fps) or fps<=0 or not np.isfinite(early_seconds) or early_seconds<=0):
        raise ValueError('Finite matched poses and positive explicit clock required')
    if any(i<0 or i>=q.shape[1] for i in (pelvis,*palms)):
        raise ValueError('Body index outside saved state')
    def yaw(poses):
        forward=Rotation.from_quat(poses[:,pelvis,3:]).apply([1.,0.,0.])
        return np.arctan2(forward[:,1],forward[:,0])
    error=yaw(r)-yaw(q)
    heading=np.arctan2(np.sin(error),np.cos(error))
    xy=np.linalg.norm(q[:,pelvis,:2]-r[:,pelvis,:2],axis=-1)
    n=min(len(q),int(round(early_seconds*fps)))
    if n<1:raise ValueError('Diagnostic window contains no samples')
    low=np.flatnonzero(q[:,pelvis,2]<.35)
    drift=np.flatnonzero(xy>.5)
    return dict(early_seconds=n/fps,root_xy_rmse_m=float(np.sqrt(np.mean(xy[:n]**2))),
        heading_rmse_deg=float(np.degrees(np.sqrt(np.mean(heading[:n]**2)))),
        early_palm_rmse_m=float(np.sqrt(np.mean(np.sum((q[:n,palms,:3]-r[:n,palms,:3])**2,axis=-1)))),
        first_half_metre_drift_s=int(drift[0])/fps if len(drift) else None,
        first_low_pelvis_s=int(low[0])/fps if len(low) else None,
        survival_s=int(low[0])/fps if len(low) else len(q)/fps,
        samples={str(s):dict(root_xy_error_m=float(xy[min(int(s*fps),len(q)-1)]),
            heading_error_deg=float(np.degrees(heading[min(int(s*fps),len(q)-1)]))) for s in (1,3,5,10)})
