from dataclasses import asdict
import json
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.reconstruction_repair.bundle import bundle_copy, bundle_read, file_sha256, json_atomic
from real2sim2real.reconstruction_repair.cli import main
from real2sim2real.reconstruction_repair.repairs import PROPOSERS
from conftest import diagnostic_repair_run as repair_run

from conftest import box_geometry, observations, only


def test_rejected_real_motion_falls_back_byte_for_byte(make_bundle, tmp_path):
    g = box_geometry()
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g)
    r = repair_run(b.root, tmp_path / "run", only("rotation_spikes"))
    assert r.report["status"] == "unchanged_with_review_items"
    assert file_sha256(r.output_bundle / "geometry.npz") == file_sha256(b.root / "geometry.npz")
    assert file_sha256(r.output_bundle / "human.npz") == file_sha256(b.root / "human.npz")
    assert r.report["requires_optimizer_restart"] is False


def test_accepted_repair_portable_with_scores_and_restart_invalidation(make_bundle, tmp_path):
    g = box_geometry()
    o = observations(g)
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g, o=o)
    source_hash = file_sha256(b.root / "geometry.npz")
    r = repair_run(b.root, tmp_path / "run", only("rotation_spikes"))
    assert r.report["requires_optimizer_restart"]
    assert r.report["optimizer_start_update"] == 0
    assert r.report["scores"]["after_minus_before"]["silhouette_iou"]["mean"] > 0
    assert (tmp_path / "run" / "per_frame.csv").read_text().count("\n") == 10
    assert file_sha256(b.root / "geometry.npz") == source_hash
    exported = bundle_copy(r.output_bundle, tmp_path / "consumer" / "initializer")
    shutil.rmtree(b.root)
    shutil.rmtree(tmp_path / "run")
    final = bundle_read(exported.root)
    np.testing.assert_allclose(final.geometry["object_rotation"][4], np.eye(3), atol=1e-12)
    assert file_sha256(final.root / "baseline_geometry.npz") == source_hash


def test_interruption_resumes_complete_stages_and_rejects_changed_policy(make_bundle, tmp_path, monkeypatch):
    b = make_bundle()
    cfg = only("topology", "scale")
    real = PROPOSERS["scale"]
    def crash(*args):
        raise RuntimeError("synthetic process interruption")
    monkeypatch.setitem(PROPOSERS, "scale", crash)
    with pytest.raises(RuntimeError):
        repair_run(b.root, tmp_path / "run", cfg)
    cp = tmp_path / "run" / "checkpoints" / "00_topology.npz"
    stamp = cp.stat().st_mtime_ns
    monkeypatch.setitem(PROPOSERS, "scale", real)
    repaired = repair_run(b.root, tmp_path / "run", cfg, resume=True)
    assert cp.stat().st_mtime_ns == stamp
    assert bundle_read(repaired.output_bundle)
    with pytest.raises(ValueError, match="Resume mismatch"):
        repair_run(b.root, tmp_path / "run", only("topology"), resume=True)
    assert json.loads((tmp_path / "run" / "status.json").read_text())["status"] == "complete"


def test_corrupt_checkpoint_cannot_be_resumed(make_bundle, tmp_path):
    b = make_bundle()
    cfg = only()
    repair_run(b.root, tmp_path / "run", cfg)
    cp = tmp_path / "run" / "checkpoints" / "01_scale.npz"
    cp.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        repair_run(b.root, tmp_path / "run", cfg, resume=True)


def test_no_overwrite_or_nested_input(make_bundle, tmp_path):
    b = make_bundle()
    with pytest.raises(ValueError, match="non-nested"):
        repair_run(b.root, b.root / "run")
    repair_run(b.root, tmp_path / "run", only())
    with pytest.raises(ValueError, match="exists"):
        repair_run(b.root, tmp_path / "run", only())


def test_diagnostic_pipeline_is_noop_on_correct_static_scene(make_bundle, tmp_path):
    b = make_bundle()
    r = repair_run(b.root, tmp_path / "run")
    assert r.report["status"] == "unchanged_with_review_items"
    assert all(s["outcome"] == "unchanged" for s in r.report["stages"])


def test_unseen_sequence_name_and_sparse_frame_ids_are_not_selectors(make_bundle, tmp_path):
    g = box_geometry()
    o = observations(g)
    g["frame_ids"] = np.arange(9) * 12 + 5010
    g["object_rotation"][4] = Rotation.from_euler("z", 75, degrees=True).as_matrix()
    b = make_bundle(g=g, o=o)
    b.manifest["sequence_id"] = "new/unrelated-camera"
    json_atomic(b.root / "manifest.json", b.manifest)
    r = repair_run(b.root, tmp_path / "run", only("rotation_spikes"))
    accepted = [s for s in r.report["stages"] if s["outcome"] == "accepted"]
    assert accepted[0]["details"]["detected_frame_ids"] == [5058]


def test_cli_full_workflow(make_bundle, tmp_path):
    b = make_bundle()
    config = tmp_path / "config.json"
    json_atomic(config, asdict(only()))
    assert main(["validate", str(b.root)]) == 0
    assert main(["run", "--input", str(b.root), "--output", str(tmp_path / "run"), "--config", str(config)]) == 0
    assert main(["score", "--before", str(b.root), "--after", str(tmp_path / "run" / "bundle"),
                 "--output", str(tmp_path / "scores.json")]) == 0
    assert main(["export", "--input", str(tmp_path / "run" / "bundle"), "--output", str(tmp_path / "portable")]) == 0
    assert main(["validate", str(tmp_path / "portable")]) == 0


def test_two_steps_cannot_accumulate_contact_regression(make_bundle, tmp_path, monkeypatch):
    from real2sim2real.reconstruction_repair.contracts import RepairProposal
    true = box_geometry()
    o = observations(true)
    o["hand_points_world"][:] = [.168, 0, 2.]
    g = {k: v.copy() for k, v in true.items()}
    g["object_translation"][:, 0] = .008
    b = make_bundle(g=g, o=o)
    def step(name):
        def propose(bundle, parent, config):
            candidate = {k: v.copy() for k, v in parent.items()}
            candidate["object_translation"][:, 0] -= .004
            return RepairProposal(name, candidate, "Controlled pair of individually passing geometric proposals")
        return propose
    monkeypatch.setitem(PROPOSERS, "scale", step("scale"))
    monkeypatch.setitem(PROPOSERS, "pose_recovery", step("pose_recovery"))
    r = repair_run(b.root, tmp_path / "run", only("scale", "pose_recovery"))
    assert r.report["aggregate_validation"] is not None
    assert "hand_surface_separation_regressed" in r.report["aggregate_validation"]["reasons"]
    assert r.report["status"] == "unchanged_with_review_items"
    assert file_sha256(r.output_bundle / "geometry.npz") == file_sha256(b.root / "geometry.npz")
    assert len([s for s in r.report["stages"] if s["outcome"] == "superseded_by_final_fallback"]) == 2
