import json
import shutil

import numpy as np
import pytest

from real2sim2real.reconstruction_repair.bundle import (arrays_atomic, bundle_copy, bundle_read, file_sha256,
                                             geometry_validate, json_atomic)
from real2sim2real.reconstruction_repair.cli import main
from real2sim2real.reconstruction_repair.contracts import RepairConfig

from conftest import box_geometry, metadata


@pytest.mark.parametrize("failure", ["reflection", "wrong_units", "nan", "time_reverse", "frame_duplicates", "bad_faces", "camera_scale"])
def test_invalid_geometry_rejected(make_bundle, failure):
    g, m = box_geometry(), metadata()
    if failure == "reflection":
        g["object_rotation"][3, 0, 0] = -1
    elif failure == "wrong_units":
        m["coordinates"] = {**m["coordinates"], "length_unit": "cm"}
    elif failure == "nan":
        g["vertices"][0, 1] = np.nan
    elif failure == "time_reverse":
        g["timestamps_s"][2] = -1
    elif failure == "frame_duplicates":
        g["frame_ids"][2] = g["frame_ids"][1]
    elif failure == "bad_faces":
        g["faces"][0, 1] = 999
    else:
        g["world_to_camera"][0, 0, 0] = 2
    with pytest.raises(ValueError):
        make_bundle(g=g, m=m, o={})


def test_corrupt_or_external_assets_rejected(make_bundle, tmp_path):
    b = make_bundle()
    (b.root / "geometry.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        bundle_read(b.root)
    c = make_bundle()
    outside = tmp_path / "external.npz"
    shutil.copyfile(c.root / "geometry.npz", outside)
    (c.root / "geometry.npz").unlink()
    (c.root / "geometry.npz").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        bundle_read(c.root)
    c.manifest["assets"]["geometry"]["path"] = "../external.npz"
    json_atomic(c.root / "manifest.json", c.manifest)
    with pytest.raises(ValueError, match="inside"):
        bundle_read(c.root)


def test_export_survives_source_removal(make_bundle, tmp_path):
    b = make_bundle()
    exported = bundle_copy(b.root, tmp_path / "standalone")
    expected = file_sha256(b.root / "human.npz")
    shutil.rmtree(b.root)
    checked = bundle_read(exported.root)
    assert file_sha256(checked.root / "human.npz") == expected
    assert not any(p.is_symlink() for p in checked.root.rglob("*"))


def test_config_typos_and_wrong_types_fail():
    for values in ({"translation_bound_cm": 2}, {"pose_recovery": "false"}, {"max_fit_anchors": 1.5},
                   {"max_gap_s": float("nan")}, {"translation_bound_m": -1}):
        with pytest.raises(ValueError):
            RepairConfig.from_dict(values)


def test_small_inherited_rotation_roundoff_is_preserved_but_scaling_is_rejected(make_bundle):
    g = box_geometry()
    g["object_rotation"][0] *= 1.00001
    b = make_bundle(g=g)
    np.testing.assert_array_equal(b.geometry["object_rotation"], g["object_rotation"])
    g["object_rotation"][0] *= 1.001
    with pytest.raises(ValueError):
        geometry_validate(g)


def test_numeric_archive_import_preserves_native_human(make_bundle, tmp_path):
    b = make_bundle()
    g = b.geometry
    raw = {k: g[k] for k in ("vertices", "faces", "object_scale", "object_rotation", "object_translation")}
    raw.update(frames=g["frame_ids"], pose=np.ones((9, 136)), shape=np.arange(45.), scales=np.zeros(68))
    arrays_atomic(tmp_path / "raw.npz", raw)
    arrays_atomic(tmp_path / "camera.npz", {"intrinsics": g["intrinsics"][0], "world_to_camera": np.eye(4)})
    json_atomic(tmp_path / "metadata.json", metadata())
    assert main(["import-npz", "--source", str(tmp_path / "raw.npz"), "--camera", str(tmp_path / "camera.npz"),
                 "--metadata", str(tmp_path / "metadata.json"), "--fps", "30", "--output", str(tmp_path / "imported")]) == 0
    imported = bundle_read(tmp_path / "imported")
    np.testing.assert_array_equal(imported.geometry["frame_ids"], g["frame_ids"])
    np.testing.assert_array_equal(imported.geometry["timestamps_s"], g["frame_ids"] / 30)
    with np.load(imported.root / "human.npz", allow_pickle=False) as h:
        np.testing.assert_array_equal(h["pose"], raw["pose"])


def test_unknown_source_field_not_silently_lost(tmp_path):
    # Import rejects source fields before any output can be mistaken for complete.
    g = box_geometry()
    raw = {k: g[k] for k in ("vertices", "faces", "object_scale", "object_rotation", "object_translation")}
    raw.update(frames=g["frame_ids"], unknown=np.ones(1))
    arrays_atomic(tmp_path / "raw.npz", raw)
    arrays_atomic(tmp_path / "camera.npz", {"intrinsics": g["intrinsics"][0], "world_to_camera": np.eye(4)})
    json_atomic(tmp_path / "meta.json", metadata())
    assert main(["import-npz", "--source", str(tmp_path / "raw.npz"), "--camera", str(tmp_path / "camera.npz"),
                 "--metadata", str(tmp_path / "meta.json"), "--fps", "30", "--output", str(tmp_path / "output")]) == 2
    assert not (tmp_path / "output").exists()
