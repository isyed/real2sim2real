"""Warm-start an anchored window with proposals from declared fitting frames.

Pose proposals are not observations. Held-out frames cannot supply initialization
poses; they are interpolated and assessed separately. The final decision is made
against the original anchor-only hypothesis, including its image preservation,
displacement and motion limits. A successful CPU result still needs GPU checks.
"""
from dataclasses import asdict
import numpy as np
from .anchor_hypothesis import initialize
from .temporal_pose import TemporalPoseConfig, validate, measurements, acceptance, refine_window


def initialize_fit_poses(data, anchor_ids, anchor_poses, proposal_ids, proposal_poses):
    times = np.asarray(data['timestamps_s'])
    delta = np.diff(times)
    if not len(delta) or delta[0] <= 0 or not np.allclose(delta, delta[0]):
        raise ValueError('Uniform explicit source timestamps required')
    fps = 1. / delta[0]
    original, anchor_receipt = initialize(data['frame_ids'], anchor_ids, anchor_poses, fps=fps)
    window = dict(data, poses_camera=original, locked=np.isin(data['frame_ids'], anchor_ids))
    validate(window)
    ids, poses = np.asarray(proposal_ids), np.asarray(proposal_poses, float)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids)) != len(ids) or poses.shape != (len(ids), 4, 4)
            or not np.isin(ids, data['frame_ids'][data['fit'].astype(bool)]).all()):
        raise ValueError('Unique declared fit-frame pose proposals required; held-out poses forbidden')
    anchors = {int(i): p for i, p in zip(anchor_ids, anchor_poses)}
    if any(int(i) in anchors and not np.array_equal(p, anchors[int(i)]) for i, p in zip(ids, poses)):
        raise ValueError('A fitting proposal cannot change a registered anchor')
    merged = {int(i): p for i, p in zip(ids, poses)}
    merged.update(anchors)
    knots = np.array(sorted(merged), dtype=int)
    seeded, receipt = initialize(data['frame_ids'], knots, np.stack([merged[int(i)] for i in knots]), fps=fps)
    receipt.update(proposal_fit_frame_ids=ids.tolist(), heldout_poses_used=False,
                   registered_anchor_ids=list(map(int, anchor_ids)), additional_proposals_are_locked=False)
    return window, seeded, dict(anchor_only=anchor_receipt, dense=receipt)


def refine(data, anchor_ids, anchor_poses, proposal_ids, proposal_poses, *,
           config=TemporalPoseConfig(benefit_mode='observation')):
    config.validate()
    baseline, seeded, receipt = initialize_fit_poses(data, anchor_ids, anchor_poses, proposal_ids, proposal_poses)
    original = baseline['poses_camera']
    before = measurements(baseline, original, config)
    after = measurements(baseline, seeded, config)
    initial_decision = acceptance(baseline, before, after, original, seeded, config)
    optimizer = None
    if initial_decision['accepted']:
        candidate = seeded.copy()
    else:
        candidate, optimizer = refine_window(dict(baseline, poses_camera=seeded), config)
        after = measurements(baseline, candidate, config)
    decision = acceptance(baseline, before, after, original, candidate, config)
    return candidate, dict(schema='real2sim2real.dense-anchored-recovery/1',
        initialization=receipt, initial_decision=initial_decision, optimizer=optimizer,
        before=before, after=after, decision=decision, config=asdict(config),
        thresholds_relaxed=False, heldout_poses_used_for_initialization=False,
        comparison_baseline='unchanged_registered_anchor_only_hypothesis',
        requires_gpu_validation=True, full_reference_accepted=False, physics_validated=False)
