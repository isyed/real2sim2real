import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, load
from real2sim2real.rss_bridge.bundle import create_bundle, import_poses, read_geometry, rigid
from real2sim2real.rss_bridge.scene import camera_parameters


@pytest.fixture
def inputs(tmp_path):
    n = 4
    rotation = Rotation.from_euler("xyz", [[12, -17, 4], [11, 8, 27],
                                          [9, 12, 23], [20, 3, 9]], degrees=True).as_matrix()
    extrinsic = np.tile(np.eye(4), (n, 1, 1))
    extrinsic[:, :3, :3] = Rotation.from_euler("x", 0.2).as_matrix()
    extrinsic[:, :3, 3] = [0.3, -0.4, 0.7]
    values = dict(vertices=np.array([[0., 0., 0.], [0.1, 0., 0.], [0., 0.2, 0.]]),
                  faces=np.array([[0, 1, 2]]), object_rotation=rotation,
                  object_translation=np.array([[0., 0., 2.], [.1, .2, 2.1],
                                               [.2, .3, 2.1], [.4, .2, 2.2]]),
                  object_scale=np.array(2.5), intrinsics=np.tile(
                      np.array([[510., 0., 321.], [0., 525., 222.], [0., 0., 1.]]), (n, 1, 1)),
                  world_to_camera=extrinsic, frame_ids=np.arange(n) * 2,
                  timestamps_s=np.arange(n) / 15, untouched_extra=np.array([123]))
    geometry = tmp_path / "geometry.npz"
    np.savez_compressed(geometry, **values)
    mesh = tmp_path / "original.obj"
    mesh.write_text("v 0 0 0\nv .1 0 0\nv 0 .2 0\nf 1 2 3\n")
    human = tmp_path / "human.npz"
    np.savez(human, opaque=np.array([3, 4, 5]))
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.2]).as_matrix()
    transform[:3, 3] = [.3, .7, -.2]
    bundle = create_bundle(geometry, mesh, human, tmp_path / "exchange",
                           source_to_rss=transform, width=640, height=480,
                           source_frame="test_world", rss_revision="pinned-test",
                           human_coordinate_frame="independent_human_frame")
    return bundle, geometry, transform


def test_identity_roundtrip_is_byte_exact(inputs, tmp_path):
    bundle, geometry, _ = inputs
    result = import_poses(bundle, bundle / "poses_rss.npz", tmp_path / "returned")
    assert digest(result / "geometry.npz") == digest(geometry)
    assert digest(result / "human.npz") == digest(bundle / "human.npz")
    assert load(result)["metadata"]["physical_validation"] is False
    assert json.loads((result / "edit_receipt.json").read_text())["changed_frame_ids"] == []


def test_edit_returns_to_source_frame_and_preserves_other_data(inputs, tmp_path):
    bundle, geometry, transform = inputs
    with np.load(bundle / "poses_rss.npz") as saved:
        edits = dict(saved)
    displacement = np.array([.02, -.03, .04])
    edits["object_to_rss"][1, :3, 3] += transform[:3, :3] @ displacement
    np.savez(tmp_path / "edit.npz", **edits)
    result = import_poses(bundle, tmp_path / "edit.npz", tmp_path / "returned")
    old, new = read_geometry(geometry), read_geometry(result / "geometry.npz")
    np.testing.assert_allclose(new["object_translation"][1] - old["object_translation"][1],
                               displacement, atol=1e-14, rtol=0)
    for key in old:
        if key not in {"object_translation", "object_rotation"}:
            np.testing.assert_array_equal(old[key], new[key])
    np.testing.assert_array_equal(old["object_rotation"][[0, 2, 3]],
                                  new["object_rotation"][[0, 2, 3]])
    assert json.loads((result / "edit_receipt.json").read_text())["changed_frame_ids"] == [2]


@pytest.mark.parametrize("defect", ["missing", "reordered", "retimed", "scale", "reflection", "nan", "extra"])
def test_invalid_edits_rejected(inputs, tmp_path, defect):
    bundle, _, _ = inputs
    with np.load(bundle / "poses_rss.npz") as saved:
        edits = dict(saved)
    if defect == "missing":
        edits = {k: v[:-1] for k, v in edits.items()}
    elif defect == "reordered":
        edits = {k: v[::-1] for k, v in edits.items()}
    elif defect == "retimed":
        edits["timestamps_s"][1] += 0.001
    elif defect == "scale":
        edits["object_to_rss"][1, :3, :3] *= 1.01
    elif defect == "reflection":
        edits["object_to_rss"][1, :3, 0] *= -1
    elif defect == "nan":
        edits["object_to_rss"][1, 0, 3] = np.nan
    else:
        edits["new_mesh"] = np.zeros(3)
    np.savez(tmp_path / "bad.npz", **edits)
    with pytest.raises(ValueError):
        import_poses(bundle, tmp_path / "bad.npz", tmp_path / "returned")
    assert not (tmp_path / "returned").exists()


def test_tampered_human_rejected(inputs, tmp_path):
    bundle, _, _ = inputs
    (bundle / "human.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        import_poses(bundle, bundle / "poses_rss.npz", tmp_path / "returned")


def test_existing_output_is_never_replaced(inputs, tmp_path):
    bundle, _, _ = inputs
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        import_poses(bundle, bundle / "poses_rss.npz", output)
    assert (output / "sentinel").read_text() == "keep"


def test_camera_transform_uses_xyzw_and_cv_axes(inputs):
    _, geometry, transform = inputs
    data = read_geometry(geometry)
    params = camera_parameters(data["intrinsics"][0], data["world_to_camera"][0],
                               transform, width=640, height=480)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(params["xform"][3:]).as_matrix()
    matrix[:3, 3] = params["xform"][:3]
    np.testing.assert_allclose(matrix, transform @ np.linalg.inv(data["world_to_camera"][0]), atol=1e-14)
    assert [params[k] for k in ("fx", "fy", "cx", "cy")] == [510, 525, 321, 222]


def test_actual_rss_projection(inputs):
    pytest.importorskip("robosimstudio")
    from real2sim2real.rss_bridge.scene import camera_parity
    result = camera_parity(inputs[0], [0, 2, 6])
    assert result["passed"]
    assert result["physics_tested"] is False


def test_unit_transform_rejects_scaling():
    scaled = np.eye(4)
    scaled[:3, :3] *= .001
    with pytest.raises(ValueError, match="scale"):
        rigid(scaled)


def test_float32_tracking_drift_is_declared_and_identity_preserved(inputs, tmp_path):
    bundle, geometry, transform = inputs
    values = read_geometry(geometry)
    values["object_rotation"][0] *= 1 + 2e-6
    noisy = tmp_path / "noisy.npz"
    np.savez_compressed(noisy, **values)
    exchange = create_bundle(noisy, bundle / "source_mesh.obj", bundle / "human.npz",
                              tmp_path / "noisy_exchange", source_to_rss=transform,
                              width=640, height=480, source_frame="test",
                              rss_revision="test", human_coordinate_frame="human")
    metadata = load(exchange)["metadata"]
    assert metadata["source_rotation_orthogonality_max_abs"] > 2e-6
    assert 0 < metadata["rss_rotation_vertex_displacement_bound_m"] < 1e-5
    with np.load(exchange / "poses_rss.npz") as data:
        rigid(data["object_to_rss"])
    returned = import_poses(exchange, exchange / "poses_rss.npz", tmp_path / "returned")
    assert digest(returned / "geometry.npz") == digest(noisy)
