"""Linux filesystem/process isolation; never silently fall back to host execution."""

import os
import signal
import subprocess
import sys
from pathlib import Path


def command(argv, *, readonly=(), writable=(), cwd, network=False):
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all"]
    if network:
        args += ["--share-net"]
    for path in ["/usr", "/bin", "/lib", "/lib64"]:
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in [
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/etc/ssl",
        "/etc/resolv.conf",
        "/etc/hosts",
    ]:
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    # A wheel install lives inside prefix. Editable installs are not deployment.
    roots = {
        str(Path(sys.prefix).resolve()),
        *(str(Path(p).resolve()) for p in readonly),
    }
    for path in sorted(roots):
        if path not in {"/", "/usr"}:
            args += ["--ro-bind", path, path]
    for path in writable:
        path = str(Path(path).resolve())
        args += ["--bind", path, path]
    args += [
        "--chdir",
        str(Path(cwd).resolve()),
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--",
        *map(str, argv),
    ]
    return args


def environment(extra=None):
    return {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        **(extra or {}),
    }


def kill(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run(argv, *, timeout, cwd, readonly=(), network=False, extra_env=None):
    with subprocess.Popen(
        command(argv, readonly=readonly, writable=[cwd], cwd=cwd, network=network),
        cwd=cwd,
        env=environment(extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill(process)
            raise
    if process.returncode:
        raise RuntimeError(
            f"Isolated process exited {process.returncode}: {stderr[-4000:]}"
        )
    return stdout, stderr
