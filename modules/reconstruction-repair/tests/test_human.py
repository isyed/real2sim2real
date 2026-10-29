"""Procedural decoder doubles test policy; the native smoke tests real assets separately."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from real2sim2real.reconstruction_repair import BundleData, RepairConfig
from real2sim2real.reconstruction_repair.bundle import bundle_create, bundle_read, file_sha256, json_atomic
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
from real2sim2real.reconstruction_repair.human_contract import DecodedHuman, REPAIR_CONTROLS, human_parameters
from real2sim2real.reconstruction_repair.human_runner import human_preflight
from real2sim2real.reconstruction_repair.human_proposals import human_motion_proposal, human_segments
from conftest import diagnostic_repair_run as repair_run
from conftest import box_geometry, metadata, only


class FixtureDecoder:
    def __init__(self):
        self.parameter_names = ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz", *REPAIR_CONTROLS]
        self.parameter_names += [f"frozen_{i}" for i in range(len(self.parameter_names), 204)]
        self.faces = box_geometry()["faces"]
        self.hand_vertices = (np.array([0, 3]), np.array([1, 2]))
        self.identity = {"backend": "procedural-test-only", "model_sha256": "0"*64, "buffers_sha256": "1"*64,
                         "parameter_names": self.parameter_names}

    def decode(self, p, transform):
        vertices = np.broadcast_to(box_geometry()["vertices"], (len(p.pose), 8, 3)).copy()
        vertices[:, 0, 0] += p.pose[:, 6] * .01
        vertices += p.pose[:, None, :3]*.1
        joints = vertices[:, np.arange(17)%8].copy()
        return DecodedHuman(vertices @ transform[:3, :3].T + transform[:3, 3], joints @ transform[:3, :3].T + transform[:3, 3])


def human_fixture(tmp_path, *, real_motion=False, missing=None, transform=None):
    decoder = FixtureDecoder()
    g = box_geometry(9); g["object_translation"][:, 2] = 6.
    transform = np.eye(4) if transform is None else transform
    pose = np.zeros((9, 136)); pose[:, 2] = 20
    pose[4, 0] = .09
    raw = {"pose": pose, "shape": np.zeros(45), "scales": np.zeros(68)}
    original = decoder.decode(human_parameters(raw, 9), transform)
    target_pose = pose.copy()
    if not real_motion: target_pose[4, 0] = 0
    target = decoder.decode(human_parameters({**raw, "pose": target_pose}, 9), transform)
    masks = []
    temp = {**g, "object_translation": np.zeros((9, 3)), "faces": decoder.faces}
    for i in range(9):
        temp["vertices"] = target.vertices_world[i]
        masks.append(mesh_silhouette(temp, i, 96, 128))
    projected = np.einsum("tij,tkj->tki", g["intrinsics"], target.joints_world)
    o = {"human_mask": np.stack(masks), "human_reliable": np.ones(9, bool),
         "human_keypoints_xy": projected[:, :, :2]/projected[:, :, 2:],
         "human_keypoints_confidence": np.ones((9, 17)), "human_contact_active": np.zeros((9, 2), bool)}
    raw.update(parity_frame_ids=g["frame_ids"][[0, 4, 8]], parity_vertices_world=original.vertices_world[[0, 4, 8]])
    m = metadata()
    m["human"] = {"encoding": "mhr-native-136-v1", "native_to_world": transform.tolist(),
                  "decoder_assets": {k:decoder.identity[k] for k in ("model_sha256", "buffers_sha256")}, "parity_source": "procedural independent geometry fixture"}
    m["evidence"].update(human_observation_source="procedural control", human_contact_source="explicit no-contact control")
    if missing:
        if missing in raw: del raw[missing]
        else: del o[missing]
    return bundle_create(tmp_path/"input", BundleData(m, g, o, raw)), decoder


def human_config(**extra):
    values = asdict(only())
    values.update(human_repair=True, human_temporal=False)
    values.update(extra)
    return RepairConfig.from_dict(values)


def test_native_outlier_repair_and_portable_provenance(tmp_path):
    b, decoder = human_fixture(tmp_path)
    result = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    assert not result.report["human_payload_preserved"]
    out = bundle_read(result.output_bundle)
    assert out.manifest["repair"]["human_applied"]
    assert result.report["requires_optimizer_restart"]
    assert file_sha256(out.root/"geometry.npz") == file_sha256(b.root/"geometry.npz")
    assert file_sha256(out.root/"baseline_human.npz") == file_sha256(b.root/"human.npz")
    with np.load(out.root/"human.npz") as selected, np.load(b.root/"human.npz") as parent:
        assert selected["pose"][4, 0] == 0
        for key in ("shape", "scales"): np.testing.assert_array_equal(selected[key], parent[key])
        np.testing.assert_array_equal(selected["pose"][:, 3:], parent["pose"][:, 3:])
    assert (tmp_path/"run/human/per_frame.csv").exists()
    assert result.report["human_repair"]["scores"]["after_minus_before"]["joint_error_px"]["mean"] < 0


def test_real_motion_rejected_and_human_bytes_unchanged(tmp_path):
    b, decoder = human_fixture(tmp_path, real_motion=True)
    r = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    assert r.report["human_payload_preserved"]
    assert file_sha256(r.output_bundle/"human.npz") == file_sha256(b.root/"human.npz")
    assert r.report["human_repair"]["stages"][0]["outcome"] == "rejected"


@pytest.mark.parametrize("field", ["parity_vertices_world", "human_contact_active", "human_keypoints_xy"])
def test_missing_evidence_fails_before_repair(tmp_path, field):
    if field == "human_keypoints_xy":
        with pytest.raises(ValueError): human_fixture(tmp_path, missing=field)
        return
    b, decoder = human_fixture(tmp_path, missing=field)
    with pytest.raises(ValueError): human_preflight(b, decoder, human_config())


def test_decoder_parity_and_asset_identity_fail_closed(tmp_path):
    b, decoder = human_fixture(tmp_path)
    decoder.identity["model_sha256"] = "2"*64
    with pytest.raises(ValueError, match="identities"): human_preflight(b, decoder, human_config())
    decoder.identity["model_sha256"] = "0"*64
    real = decoder.decode
    def wrong(p, transform):
        result = real(p, transform); result.vertices_world += .01
        return result
    decoder.decode = wrong
    with pytest.raises(ValueError, match="parity failed"): human_preflight(b, decoder, human_config())


def test_human_double_application_is_blocked(tmp_path):
    b, decoder = human_fixture(tmp_path)
    r = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    with pytest.raises(ValueError, match="already applied"):
        repair_run(r.output_bundle, tmp_path/"repeat", human_config(), human_decoder=decoder)


def test_human_locked_contact_boundary_and_finger_preservation(tmp_path):
    b, decoder = human_fixture(tmp_path)
    b.observations["human_locked"] = np.zeros(9, np.uint8); b.observations["human_locked"][4] = 1
    raw = dict(np.load(b.root/"human.npz")); raw["pose"][4, 80] = .4
    parameters = human_parameters(raw, 9)
    proposed = human_motion_proposal(b, parameters, decoder, human_config(), "outliers")
    np.testing.assert_array_equal(proposed, parameters.pose)
    b.observations["human_locked"][:] = 0
    b.observations["human_contact_active"][4:] = True
    proposed = human_motion_proposal(b, parameters, decoder, human_config(), "temporal")
    np.testing.assert_array_equal(proposed[[0, 3, 4, 8]], parameters.pose[[0, 3, 4, 8]])
    np.testing.assert_array_equal(proposed[:, 68:], parameters.pose[:, 68:])


def test_human_stage_resume_and_corruption(tmp_path):
    b, decoder = human_fixture(tmp_path)
    r = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    cp = tmp_path/"run/human/outliers.npz"; stamp = cp.stat().st_mtime_ns
    repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder, resume=True)
    assert cp.stat().st_mtime_ns == stamp
    cp.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder, resume=True)


def test_human_disabled_does_not_load_decoder_or_change_payload(tmp_path):
    b, _ = human_fixture(tmp_path)
    r = repair_run(b.root, tmp_path/"run", only())
    assert r.report["human_repair"] is None
    assert file_sha256(r.output_bundle/"human.npz") == file_sha256(b.root/"human.npz")


def test_fit_requires_explicit_observation_holdouts(tmp_path):
    b, decoder = human_fixture(tmp_path)
    with pytest.raises(ValueError, match="fit frames"):
        human_preflight(b, decoder, human_config(human_fit=True))


def test_stale_hand_evidence_blocks_later_object_repair(tmp_path):
    b, decoder = human_fixture(tmp_path)
    r = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    with pytest.raises(ValueError, match="rebuild evidence"):
        repair_run(r.output_bundle, tmp_path/"object-repeat", only("pose_recovery"))
    copied = repair_run(r.output_bundle, tmp_path/"copy", only())
    out = bundle_read(copied.output_bundle)
    assert out.manifest["repair"]["original_hand_observations_require_refresh"]
    assert out.manifest["repair"]["human_applied"]
    assert file_sha256(out.root/"baseline_human.npz") == file_sha256(b.root/"human.npz")


def test_score_never_claims_changed_human_is_preserved(tmp_path):
    from real2sim2real.reconstruction_repair.cli import main
    b, decoder = human_fixture(tmp_path)
    r = repair_run(b.root, tmp_path/"run", human_config(), human_decoder=decoder)
    score = tmp_path/"comparison.json"
    assert main(["score", "--before", str(b.root), "--after", str(r.output_bundle), "--output", str(score)]) == 0
    data = json.loads(score.read_text())
    assert not data["human_payload_preserved"]
    assert "differ" in data["interpretation"]["human"]


def test_object_contact_regression_restores_pair_even_without_human_change(tmp_path):
    from real2sim2real.reconstruction_repair.human_runner import human_stage_run
    b, decoder = human_fixture(tmp_path)
    geometry = {k:v.copy() for k,v in b.geometry.items()}
    geometry["object_translation"][:, 2] += .1
    result = human_stage_run(b, geometry, tmp_path/"human", human_config(human_outliers=False), decoder, "test")
    assert result.report["requires_object_fallback"]
    assert not result.changed
    assert result.report["scores"]["before"] == result.report["scores"]["after"]


@pytest.mark.parametrize("changes", [{"human_confidence_min": 2}, {"human_min_joints": 13},
                                     {"human_min_joint_gain_fraction": 2}, {"human_fit": True}])
def test_human_configuration_rejects_incoherent_settings(changes):
    with pytest.raises(ValueError): RepairConfig.from_dict(changes)
