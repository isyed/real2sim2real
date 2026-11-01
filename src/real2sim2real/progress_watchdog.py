"""Bound a child process by total time and an independently observed progress token."""
import os
import signal
import subprocess
import time


def run(args, *, timeout, stall_timeout, progress, poll_seconds=2., **kwargs):
    """Terminate the complete child process group on stall, timeout or interruption.

    ``progress`` returns a comparable token (for example completed step count).
    Log noise need not count as progress. None means no progress has appeared.
    The initial compilation period is bounded by the same stall allowance.
    """
    if min(timeout,stall_timeout,poll_seconds)<=0 or kwargs.pop('shell',False):
        raise ValueError('Explicit arguments and positive finite watchdog durations required')
    import math
    if not all(math.isfinite(v) for v in (timeout,stall_timeout,poll_seconds)):
        raise ValueError('Watchdog durations must be finite')
    started=last=time.monotonic();token=None
    with subprocess.Popen(args,start_new_session=True,**kwargs) as child:
        try:
            while child.poll() is None:
                now=time.monotonic();current=progress()
                if current is not None and current!=token:
                    token=current;last=now
                if now-started>=timeout:raise TimeoutError('Child exceeded total stage time')
                if now-last>=stall_timeout:raise TimeoutError('Child made no measured progress within stall window')
                time.sleep(min(poll_seconds,max(.001,timeout-(now-started))))
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,args)
            return child.returncode
        except BaseException:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            raise
