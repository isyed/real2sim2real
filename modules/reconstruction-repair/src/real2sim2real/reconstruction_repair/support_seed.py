"""Explicit, bounded resting-support seed hypotheses; never trajectory edits."""
import numpy as np
from .metric_seed import assess_seed, compare_seeds


def propose_resting_seed(parent_vertices, parent_faces, parent_pose,
                         vertices, faces, seed, intrinsics, object_mask,
                         occluder_mask, depth_m, plane, *, resting_evidence,
                         max_translation_m=0.05, steps=5):
    """Try the smallest plane-normal move that passes parent/image/depth gates.

    The caller must provide reviewed evidence that this individual frame is a
    resting phase. No contact phase is inferred or propagated to other frames.
    A returned proposal still requires fresh tracking and physical validation.
    On rejection return the original seed unchanged, with every trial retained.
    """
    if (not isinstance(resting_evidence, str) or not resting_evidence.strip()
            or isinstance(max_translation_m, bool)
            or not np.isfinite(max_translation_m) or not 0 < max_translation_m <= .1
            or type(steps) is not int or not 1 <= steps <= 10):
        raise ValueError('Reviewed resting evidence and bounded search are required')
    initial = assess_seed(vertices, faces, seed, intrinsics, object_mask,
                          occluder_mask, depth_m, plane)
    gap = initial['minimum_support_height_m']
    direction = -np.sign(gap) * np.asarray(plane)[:3]
    moves = np.linspace(0., min(abs(gap), max_translation_m), steps + 1)
    trials = []
    selected = None
    for distance in moves:
        proposal = np.asarray(seed, float).copy()
        proposal[:3, 3] += direction * distance
        check = compare_seeds(parent_vertices, parent_faces, parent_pose,
                              vertices, faces, proposal, intrinsics, object_mask,
                              occluder_mask, depth_m, plane, require_resting=True)
        trials.append(dict(translation_m=float(distance), **check))
        if check['proposal_passed']:
            selected = proposal
            break
    return (np.asarray(seed).copy() if selected is None else selected), dict(
        schema='real2sim2real.resting-seed-proposal/1',
        candidate_available=selected is not None, trials=trials,
        resting_evidence=resting_evidence, maximum_translation_m=max_translation_m,
        original_seed_preserved_on_rejection=True, parent_modified=False,
        phase_propagated=False, requires_fresh_tracking=True,
        actor_training_approved=False, physical_contact_verified=False)
