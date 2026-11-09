from types import SimpleNamespace
import pytest

from real2sim2real.reconstruction.daid_progress import ActiveProgressClock, observe_completed_calls


def test_observer_preserves_output_identity_arguments_and_order():
    observed, calls = [], []
    token = object()
    def original(*args, **kw):
        calls.append((args, kw))
        return token
    owner = SimpleNamespace(step=original)
    saved = observe_completed_calls(owner, 'step', observed.append, interval=3)
    for i in range(7):
        assert owner.step(i, seed=42) is token
    assert saved is original
    assert observed == [1, 3, 6]
    assert calls == [((i,), {'seed': 42}) for i in range(7)]


def test_failed_step_does_not_claim_progress():
    receipts = []
    def fail():
        raise RuntimeError('step failed')
    owner = SimpleNamespace(step=fail)
    observe_completed_calls(owner, 'step', receipts.append)
    with pytest.raises(RuntimeError, match='step failed'):
        owner.step()
    assert receipts == []


def test_pause_does_not_consume_progress_allowance_and_new_log_resets():
    clock = ActiveProgressClock(0, 100)
    assert clock.sample(230, 100) == 230
    assert clock.sample(330, 100, paused=True) == 230
    assert clock.sample(335, 100) == 235
    assert clock.sample(340, 101) == 0
    assert clock.sample(581, 101) == 241


def test_backwards_clock_rejected():
    with pytest.raises(ValueError):
        ActiveProgressClock(10, None).sample(9, None)


def test_zero_interval_rejected():
    with pytest.raises(ValueError):
        observe_completed_calls(SimpleNamespace(step=lambda: None), 'step', print, interval=0)
