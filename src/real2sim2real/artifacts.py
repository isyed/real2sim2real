"""Portable stage receipts with content verification and explicit file roles."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def publish(root, kind, files, *, metadata=None):
    root = Path(root).resolve()
    if (root / "artifact.json").exists():
        raise FileExistsError(
            "Preserve existing stage artifacts; choose another output"
        )
    records = {}
    for role, value in files.items():
        if Path(value).is_symlink():
            raise ValueError("Stage files cannot be symlinks")
        path = Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError(
                "Stage files must be real files inside the output: " + str(path)
            )
        records[role] = dict(path=str(path.relative_to(root)), sha256=digest(path))
    if not records:
        raise ValueError("A stage artifact must contain files")
    payload = dict(
        schema="real2sim2real.artifact/1",
        kind=kind,
        files=records,
        metadata=metadata or {},
    )
    write_json(root / "artifact.json", payload)
    return root


def load(root, *, kinds=None):
    root = Path(root).resolve()
    value = json.loads((root / "artifact.json").read_text())
    if value.get("schema") != "real2sim2real.artifact/1" or (
        kinds and value["kind"] not in kinds
    ):
        raise ValueError("Incompatible input artifact")
    if not isinstance(value.get("files"), dict) or not value["files"]:
        raise ValueError("Artifact must contain files")
    for row in value["files"].values():
        path = root / row["path"]
        if (
            Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("Artifact path escapes its root")
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError("Artifact content changed: " + row["path"])
    return value


def role_path(root, role):
    value = load(root)
    if role not in value["files"]:
        raise ValueError("Input artifact has no " + role)
    return Path(root).resolve() / value["files"][role]["path"]
