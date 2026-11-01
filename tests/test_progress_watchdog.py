import sys
import pytest
from real2sim2real.progress_watchdog import run


def test_stalled_child_is_reaped():
    with pytest.raises(TimeoutError,match='no measured progress'):
        run([sys.executable,'-c','import time;time.sleep(30)'],
            timeout=5,stall_timeout=.15,poll_seconds=.02,progress=lambda:None)


def test_advancing_work_can_exceed_one_stall_window(tmp_path):
    p=tmp_path/'steps'
    code='import pathlib,time\np=pathlib.Path('+repr(str(p))+')\nfor i in range(8):\n p.write_text(str(i));time.sleep(.07)\n'
    assert run([sys.executable,'-c',code],timeout=4,stall_timeout=.3,poll_seconds=.02,
        progress=lambda:p.read_text() if p.exists() else None)==0


def test_progress_does_not_bypass_total_bound():
    import time
    with pytest.raises(TimeoutError,match='total stage'):
        run([sys.executable,'-c','import time;time.sleep(30)'],
            timeout=.2,stall_timeout=2,poll_seconds=.02,progress=time.monotonic)
