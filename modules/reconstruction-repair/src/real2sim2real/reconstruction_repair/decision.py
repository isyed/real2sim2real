"""Compose optimizer and independent validation decisions without promotion.

An optimizer's geometric success is only one gate. External interaction checks
may veto it, but can never turn a rejected optimizer proposal into an acceptance.
This helper records supplied measurements; it does not measure physical contact.
"""
from copy import deepcopy


def combine_decision(primary, checks):
    if (not isinstance(primary.get('accepted'), bool) or
            not isinstance(primary.get('reasons'), list) or
            any(not isinstance(reason, str) for reason in primary['reasons'])):
        raise ValueError('Require an explicit primary acceptance and reason list')
    checks = deepcopy(list(checks))
    names = []
    for check in checks:
        name = check.get('id')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Independent checks need unique nonempty IDs')
        if not isinstance(check.get('passed'), bool):
            raise ValueError('An unmeasured check cannot be treated as passing')
        names.append(name)
    result = deepcopy(primary)
    result['primary_accepted'] = primary['accepted']
    result['independent_checks'] = checks
    failed = [check['id'] for check in checks if not check['passed']]
    result['reasons'] = list(dict.fromkeys(primary['reasons'] + failed))
    result['accepted'] = bool(primary['accepted'] and not result['reasons'])
    result.update(full_reference_accepted=False, physics_validated=False,
                  actor_training_approved=False, approved_for_use=False)
    return result
