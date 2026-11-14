"""Immutable input/proposal snapshots and campaign writer exclusion."""

import fcntl
import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

from real2sim2real.artifacts import write_json


def hashes(root):
    root = Path(root)
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not portable campaign inputs")
        if path.is_file():
            values[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return values


def copy_inputs(source, destination):
    hashes(source)
    if Path(destination).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("Output cannot be inside the input tree")
    shutil.copytree(source, destination)
    return hashes(destination)


@contextmanager
def locked(root):
    with (Path(root) / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load(root):
    root = Path(root).resolve()
    state = json.loads((root / "state.json").read_text())
    if hashes(root / "bundle") != state["bundle_hashes"]:
        raise ValueError("Frozen experiment bundle changed")
    if state["config"] != json.loads((root / "bundle/experiment.json").read_text()):
        raise ValueError("Campaign configuration differs from the frozen experiment")
    for row in state["rounds"]:
        if hashes(root / "rounds" / row["id"] / "candidate") != row["candidate_hashes"]:
            raise ValueError("Archived proposal changed")
    return state


def store(root, state):
    write_json(Path(root) / "state.json", state)
