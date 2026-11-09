"""Select matched, finite evaluation cohorts without choosing a lucky trial."""
import math


def choose(baseline, candidates, *, auc_tolerance=1e-5, maximum_success_auc_loss=.02):
    """AUC first; position error is lower-is-better. Success can offset a small AUC loss.

    Inputs contain mean cohort metrics and an explicit compatibility identity.
    Callers must use the same trials, scoring frame and evaluator build. These
    thresholds are declared selection policy, not a significance claim.
    """
    keys = ('auc', 'mppe_cm', 'sp_sr', 'mp_sr')
    def valid(row):
        return (row.get('valid') is True and row.get('identity') == baseline.get('identity')
                and bool(row.get('identity')) and row.get('trials', 0) >= 3
                and all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in keys))
    if not valid(baseline):
        raise ValueError('A valid matched frozen cohort is required')
    accepted = [baseline]
    rejected = []
    for row in candidates:
        reason = None
        if not valid(row): reason = 'invalid or incompatible cohort'
        else:
            da = row['auc'] - baseline['auc']
            success = row['sp_sr'] > baseline['sp_sr'] or row['mp_sr'] > baseline['mp_sr']
            qualified = (da > auc_tolerance or
                (da >= -auc_tolerance and row['mppe_cm'] < baseline['mppe_cm']) or
                (success and da >= -maximum_success_auc_loss))
            if not qualified: reason = 'no qualifying score improvement over frozen baseline'
        if reason: rejected.append({'id': row['id'], 'reason': reason})
        else: accepted.append(row)
    # Among qualifying candidates AUC remains first, then success and MPPE.
    successes = [r for r in accepted if r['sp_sr'] > baseline['sp_sr'] or r['mp_sr'] > baseline['mp_sr']]
    pool = successes if successes and len(accepted) == 2 else accepted
    winner = max(pool, key=lambda r: (r['auc'], r['sp_sr'], r['mp_sr'], -r['mppe_cm']))
    return {'selected': winner['id'], 'rejected': rejected, 'primary_trial': 1,
            'selection_scope': 'matched repeated offline evaluations', 'training_performed': False}
