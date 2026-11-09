"""Progress receipts after completed inference steps, without changing outputs.

Keep this observation-only adapter outside the pinned upstream checkout. CUDA
synchronization belongs to the supplied callback so logs cannot count work that
has merely been queued. It does not generate artificial heartbeat messages.
"""
from functools import wraps


def observe_completed_calls(owner, name, callback, *, interval=25):
    if interval < 1:
        raise ValueError('interval must be positive')
    original = getattr(owner, name)
    calls = 0

    @wraps(original)
    def observed(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1 or calls % interval == 0:
            callback(calls)
        return result

    setattr(owner, name, observed)
    return original


class ActiveProgressClock:
    """Count time without new progress, excluding verified thermal pauses."""
    def __init__(self, now, token):
        self.last_time = now
        self.token = token
        self.idle_seconds = 0.0

    def sample(self, now, token, *, paused=False):
        if now < self.last_time:
            raise ValueError('monotonic time moved backwards')
        if token != self.token:
            self.idle_seconds = 0.0
            self.token = token
        elif not paused:
            self.idle_seconds += now - self.last_time
        self.last_time = now
        return self.idle_seconds
