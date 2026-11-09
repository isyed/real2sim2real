import sys
import pytest
from real2sim2real.progress_watchdog import run


def test_stalled_child_is_reaped():
    with pytest.raises(TimeoutError, match="no measured progress"):
        run(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            timeout=5,
            stall_timeout=0.15,
            poll_seconds=0.02,
            progress=lambda: None,
        )


def test_advancing_work_can_exceed_one_stall_window(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock
    from real2sim2real import progress_watchdog as watchdog

    # Test progress renewal without depending on OS scheduling a real child
    # within a subsecond window. The other tests exercise real process cleanup.
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        watchdog,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=sleep,
        ),
    )
    child = SimpleNamespace(
        pid=123,
        poll=lambda: 0 if clock.now >= 8 else None,
        returncode=0,
        wait=Mock(return_value=0),
    )
    popen = Mock(return_value=nullcontext(child))
    monkeypatch.setattr(watchdog.subprocess, "Popen", popen)
    terminate = Mock()
    monkeypatch.setattr(watchdog.os, "killpg", terminate)
    assert (
        run(
            ["worker"],
            timeout=20,
            stall_timeout=2.5,
            poll_seconds=1,
            progress=lambda: int(clock.now // 2),
        )
        == 0
    )
    assert clock.now == 8  # longer than three stall windows
    popen.assert_called_once_with(["worker"], start_new_session=True)
    terminate.assert_not_called()


def test_progress_does_not_bypass_total_bound():
    import time

    with pytest.raises(TimeoutError, match="total stage"):
        run(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            timeout=0.2,
            stall_timeout=2,
            poll_seconds=0.02,
            progress=time.monotonic,
        )
