"""Bounded full-reference approach learning with an explicit assistance cutoff.

This composes the tested portable-policy physics and coarse approach profile.
It never launches work, promotes a controller or substitutes a different source.
"""
from copy import deepcopy
import math

TARGETS = (5_000_000, 10_000_000, 15_000_000)


def configuration(policy, output, *, environment_steps, worlds=256,
                  replay_capacity=262144, resume_state=None, resume_replay=None,
                  prior_configuration=None):
    from .long_training import configuration as long_configuration
    from .approach_profile import configure
    if environment_steps not in TARGETS or type(environment_steps) is not int:
        raise ValueError('Bounded approach pilot permits only 5M/10M/15M targets')
    if type(replay_capacity) is not int or replay_capacity < 1:
        raise ValueError('Explicit positive finite replay capacity required')
    c = long_configuration(policy, output, environment_steps=environment_steps,
                           worlds=worlds, resume_state=resume_state,
                           resume_replay=resume_replay, orientation_weight=.1)
    replay = c['training']['replay']
    if replay_capacity < max(replay['minimum_size'], replay['batch_size'], worlds):
        raise ValueError('Replay cannot hold warmup, sample batch and one collection')
    replay['capacity'] = replay_capacity
    c, profile = configure(c, prior_configuration=prior_configuration)
    if c['training']['resume']:
        if prior_configuration is None:
            raise ValueError('Actual prior checkpoint configuration is required')
        # Preserve the same learner's numerical recipe across bounded stops.
        # Only the transition ceiling and exact checkpoint input paths may change.
        prior = deepcopy(prior_configuration['training'])
        current = deepcopy(c['training'])
        for node in (prior, current):
            for name in ('total_environment_steps', 'resume', 'checkpoint_path'):
                node.pop(name, None)
            node['checkpoint'].pop('load_replay_path', None)
        if prior != current:
            raise ValueError('Learner/replay/optimization recipe changed during continuation')
    return c, dict(profile=profile, replay_capacity=replay_capacity,
                   checkpoints=list(TARGETS), maximum_environment_steps=TARGETS[-1],
                   assistance_zero_from=c['curriculum']['thresholds'][2],
                   evaluation_assistance=0., full_reference=True,
                   submission_replacement_automatic=False)


def diagnostic_review(baseline, current, *, target):
    """Early checkpoints diagnose progress; invalid/severe regression can stop.

    AUC is evaluated against the same declared reference as the comparator.
    The full 15M pilot ends at its ceiling regardless of improvement. Future
    training and submission selection remain separate decisions.
    """
    if target not in TARGETS:
        raise ValueError('Undeclared checkpoint')
    required = ('auc', 'mppe_cm', 'survival_s', 'upright_palm_rmse_first4s_m',
                'upright_intended_contact_s')
    valid = bool(current.get('valid')) and all(
        isinstance(current.get(k), (int, float)) and math.isfinite(current[k])
        for k in required)
    if not valid:
        return dict(continue_training=False, reason='invalid_execution', progress=False)
    score_severe = current['auc'] < baseline['auc'] - .05
    balance_severe = current['survival_s'] < max(0., baseline['survival_s'] - 3.)
    severe = score_severe and balance_severe
    progress = (current['auc'] >= baseline['auc'] + .005 or
                (current['mppe_cm'] <= baseline['mppe_cm'] * .95 and
                 current['auc'] >= baseline['auc'] - .002) or
                current['survival_s'] >= baseline['survival_s'] + 1. or
                current['upright_intended_contact_s'] > baseline['upright_intended_contact_s'] or
                current['upright_palm_rmse_first4s_m'] <=
                baseline['upright_palm_rmse_first4s_m'] * .9)
    return dict(continue_training=target < TARGETS[-1] and not severe,
                reason='severe_score_and_balance_regression' if severe else
                'bounded_15m_ceiling_reached' if target == TARGETS[-1] else
                'diagnostic_before_first_15m_gate', progress=bool(progress),
                severe_regression=bool(severe),
                diagnostic_only=target < TARGETS[-1],
                future_training_requires_separate_decision=True)
