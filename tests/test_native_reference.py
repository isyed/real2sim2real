import pytest

from real2sim2real.artifacts import digest, publish
from real2sim2real.native_reference import copy_native_reference


def source(tmp_path, tracked_cache=False):
    root = tmp_path / "source"
    root.mkdir()
    (root / "reference.parquet").write_bytes(b"native source")
    (root / "control_reference.parquet").write_bytes(b"cached control")
    (root / "control_reference.timing.json").write_text("{}")
    (root / "nested").mkdir()
    (root / "nested/control_reference.parquet").write_bytes(b"keep nested")
    files = {"reference": root / "reference.parquet"}
    if tracked_cache:
        files["playback"] = root / "control_reference.parquet"
    publish(root, "robot-reference", files)
    return root


def test_derived_cache_omitted_receipt_and_nested_assets_preserved(tmp_path):
    src = source(tmp_path)
    result = copy_native_reference(src, tmp_path / "copy")
    assert digest(src / "artifact.json") == digest(result / "artifact.json")
    assert not (result / "control_reference.parquet").exists()
    assert (result / "nested/control_reference.parquet").is_file()
    (result / "reference.parquet").write_bytes(b"new")
    assert (src / "reference.parquet").read_bytes() == b"native source"
    assert (src / "control_reference.parquet").is_file()


def test_tracked_cache_never_discarded(tmp_path):
    src = source(tmp_path, tracked_cache=True)
    with pytest.raises(ValueError, match="tracked playback"):
        copy_native_reference(src, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_reference_symlink_rejected(tmp_path):
    src = source(tmp_path)
    (src / "link").symlink_to(src / "reference.parquet")
    with pytest.raises(ValueError, match="symlinks"):
        copy_native_reference(src, tmp_path / "copy")


def test_nested_output_rejected(tmp_path):
    src = source(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        copy_native_reference(src, src / "copy")
