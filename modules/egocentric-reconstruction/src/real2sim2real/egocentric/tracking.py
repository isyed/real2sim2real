"""Bounded tracking retries that never seed from a rejected hypothesis."""

import numpy as np


def refinement_schedule(primary=5, retries=()):
    values = [primary, *retries]
    if len(values) > 4 or len(set(values)) != len(values):
        raise ValueError("Use at most three distinct retry iteration counts")
    if any(
        isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10
        for n in values
    ):
        raise ValueError("Refinement iteration counts must be integers from 1 to 10")
    return values


def refine_with_retries(seed, propose, assess, restore, *, primary=5, retries=()):
    """Try a fixed schedule under the same acceptance test.

    ``propose`` may mutate tracker state. Every retry restores the last accepted
    seed first. Accepted proposals keep their state; total failure or an exception
    rolls back. Return the last candidate and all attempts for diagnostic output.
    No reference trajectory is used or accepted by this interface.
    """
    schedule = refinement_schedule(primary, retries)
    seed = np.asarray(seed).copy()
    attempts = []
    try:
        for index, iterations in enumerate(schedule):
            if index:
                restore(seed.copy())
            pose = np.asarray(propose(iterations)).copy()
            quality = assess(pose)
            attempts.append(
                dict(iterations=iterations, pose=pose.tolist(), quality=quality.copy())
            )
            if quality["accepted"]:
                return pose, quality, attempts
    except BaseException:
        restore(seed.copy())
        raise
    restore(seed.copy())
    return pose, quality, attempts
