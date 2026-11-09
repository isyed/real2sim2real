"""Explicit reset-distribution amendments for an already unassisted learner.

This preserves rewards and the learner/replay state. It is a declared curriculum
change, not bit-identical continuation. Recovery resets retain their configured
settling assistance; immediate frame-zero starts do not use that assistance.
"""
from copy import deepcopy
import math


def amend_reset_curriculum(config, *, boundaries, probabilities):
    """Return a copy with absolute, collection-aligned reset stage boundaries.

    Each boundary ends a stage; the final stage remains active beyond its end.
    Callers must separately enforce the authorized total transition ceiling.
    """
    c = deepcopy(config)
    worlds = c['scene']['world_count']
    if len(boundaries) != len(probabilities) or not boundaries:
        raise ValueError('One end boundary per reset probability required')
    if any(type(x) is not int or x < worlds for x in boundaries):
        raise ValueError('Positive integer absolute transition boundaries required')
    rounded = [x // worlds * worlds for x in boundaries]
    if any(a >= b for a, b in zip(rounded, rounded[1:])):
        raise ValueError('Boundaries must increase after collection rounding')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Reset probabilities must be finite and within [0,1]')
    if any(a > b for a, b in zip(probabilities, probabilities[1:])):
        raise ValueError('Beginning-start probability must not decrease')
    template = c['curriculum']['stages'][-1]
    if template['voc_scale'] != 0:
        raise ValueError('Only continue an already zero-task-assistance curriculum')
    stages = []
    for p in probabilities:
        stage = deepcopy(template)
        stage.update(voc_scale=0.0, immediate_first_frame_probability=float(p))
        stages.append(stage)
    c['curriculum'].update(thresholds=rounded, stages=stages)
    c['training']['curriculum'] = deepcopy(c['curriculum'])
    return c


def continuation_review(incumbent, previous, current, *, major_review=False):
    """Repeated-regression stop and a broad major-review progress gate.

    Metrics are repeated rollout means. Contact is diagnostic, not task success;
    callers supply task-appropriate contact counts (hand counts cannot grade a
    foot push). This does not select a competition submission.
    """
    keys = ('auc', 'mppe_cm', 'first_low_pelvis_frame')
    if not current.get('valid') or any(not math.isfinite(current[k]) for k in keys):
        return dict(continue_training=False, reason='invalid_evaluation')

    def severe(m):
        return (m['auc'] < incumbent['auc'] - .05 or
                m['first_low_pelvis_frame'] < incumbent['first_low_pelvis_frame'] - 150)

    def progress(m):
        return (m['auc'] >= incumbent['auc'] + .005 or
                (m['mppe_cm'] <= incumbent['mppe_cm'] * .95 and
                 m['auc'] >= incumbent['auc'] - .002) or
                (any(m.get(k, 0) > incumbent.get(k, 0) for k in ('sp_sr', 'mp_sr')) and
                 m['auc'] >= incumbent['auc'] - .03) or
                (m['first_low_pelvis_frame'] >= incumbent['first_low_pelvis_frame'] + 50 and
                 m['auc'] >= incumbent['auc'] - .002))

    repeated_regression = severe(current) and previous is not None and severe(previous)
    stalled_major_review = (major_review and previous is not None and
                            not progress(current) and not progress(previous))
    stop = repeated_regression or stalled_major_review
    return dict(continue_training=not stop, repeated_severe_regression=repeated_regression,
                major_review=major_review, current_progress=progress(current),
                reason=('repeated_severe_regression' if repeated_regression else
                        'no_progress_at_major_review' if stalled_major_review else
                        'continue_bounded_experiment'))
