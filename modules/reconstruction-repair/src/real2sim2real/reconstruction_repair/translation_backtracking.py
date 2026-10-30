"""Bounded smaller translation proposals under unchanged repair gates.

This standalone API does not depend on optional local-window assembly tools.
"""
import numpy as np
from .continuation import _poses


def backtrack_translation(data, proposal, config, *, fractions=(.5, .25, .125, .0625),
                          editable=None, extra_gate=None):
    """Try smaller translation corrections under the unchanged window gates.

    Optional editable frames restrict changes; all other poses stay exact.
    An extra interaction gate may reject, never override the built-in gates.
    Selection checks are development validation, not an unseen test set.
    """
    from .temporal_pose import validate, measurements, acceptance
    config.validate();validate(data)
    original = data['poses_camera']
    proposal, _ = _poses(proposal, data['frame_ids'])
    if not np.array_equal(proposal[:, :3, :3], original[:, :3, :3]):
        raise ValueError('Translation backtracking cannot change rotations')
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any(fractions <= 0) or np.any(fractions >= 1)
            or np.any(np.diff(fractions) >= 0)):
        raise ValueError('Require strictly decreasing fractions inside (0,1)')
    editable = np.ones(len(original), dtype=bool) if editable is None else np.asarray(editable)
    if editable.shape != (len(original),) or editable.dtype.kind != 'b':
        raise ValueError('Editable-frame flags must be a boolean vector')
    editable = editable & ~data['locked'].astype(bool)
    before = measurements(data, original, config)
    trials = []
    for fraction in fractions:
        candidate = original.copy()
        candidate[editable, :3, 3] += fraction*(proposal[editable, :3, 3]-original[editable, :3, 3])
        after = measurements(data, candidate, config)
        decision = acceptance(data, before, after, original, candidate, config)
        interaction = None
        if decision['accepted'] and extra_gate is not None:
            interaction = extra_gate(candidate)
            if interaction.get('passed') is not True:
                decision.update(accepted=False, reasons=decision['reasons']+['extra_interaction_gate'])
        trials.append(dict(fraction=float(fraction), decision=decision, extra_gate=interaction,
                           before=before, after=after))
        if decision['accepted']:
            return candidate, dict(accepted=True, selected_fraction=float(fraction), trials=trials,
                                   editable_frame_ids=data['frame_ids'][editable].tolist(),
                                   full_reference_accepted=False, physics_validated=False, approved_for_use=False)
    return original.copy(), dict(accepted=False, selected_fraction=None, trials=trials,
                                 editable_frame_ids=data['frame_ids'][editable].tolist(),
                                 original_retained=True, full_reference_accepted=False,
                                 physics_validated=False, approved_for_use=False)

