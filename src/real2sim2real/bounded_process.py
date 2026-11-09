"""Run a finite stage and reap its descendants on timeout or interruption."""
import os
import signal
import subprocess


def run(args, *, timeout, **kwargs):
    if timeout <= 0 or kwargs.pop('shell', False):
        raise ValueError('Use explicit arguments and a positive timeout')
    with subprocess.Popen(args, start_new_session=True, **kwargs) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, args)
    return returncode
