"""Copy a native robot reference without untracked controller playback caches."""
from pathlib import Path
import shutil

from .artifacts import load


PLAYBACK_CACHES = frozenset({
    "control_reference.parquet", "control_reference.timing.json",
    "control_reference.timing.npz",
})


def copy_native_reference(source, output):
    """Preserve the receipt and assets; regenerate derived timing downstream.

    Only the three known, untracked top-level playback caches are omitted.
    A tracked playback file is an authoritative artifact and must not be dropped.
    Source files are copied, not linked, so repackaging cannot mutate the source.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    record = load(source, kinds={"robot-reference"})
    tracked = {row["path"] for row in record["files"].values()}
    if tracked & PLAYBACK_CACHES:
        raise ValueError("Cannot omit a tracked playback artifact")
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a fresh output outside the source reference")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Reference snapshot cannot contain symlinks")

    def ignore(directory, names):
        return set(names) & PLAYBACK_CACHES if Path(directory) == source else set()

    shutil.copytree(source, output, ignore=ignore)
    load(output, kinds={"robot-reference"})
    return output
