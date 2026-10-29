import numpy as np
import pytest

from real2sim2real.artifacts import publish
from real2sim2real.reconstruction.metric import pointmap, check_pointmap
from real2sim2real.controller_actor_training.stage import overrides


def test_shared_camera_depth_roundtrip():
    k = np.array([[400.0, 0, 32], [0, 420, 24], [0, 0, 1]])
    depth = np.full((48, 64), 2.0)
    xyz = pointmap(depth, k)
    assert np.allclose(xyz[24, 32], [0, 0, 2])
    assert (
        check_pointmap(xyz, k, np.ones(depth.shape, bool))["valid_object_pixels"]
        == 3072
    )
    wrong = k.copy()
    wrong[0, 2] += 10
    with pytest.raises(ValueError, match="disagree"):
        check_pointmap(xyz, wrong, np.ones(depth.shape, bool))


def test_missing_depth_cannot_certify_size():
    k = np.eye(3)
    z = np.zeros((20, 20))
    with pytest.raises(ValueError, match="Insufficient"):
        check_pointmap(pointmap(z, k), k, np.ones(z.shape, bool))


def training_request(tmp_path):
    p = tmp_path / "reference.parquet"
    p.write_bytes(b"fixture")
    publish(tmp_path, "robot-reference", {"reference": p})
    return dict(
        input=str(tmp_path),
        output=str(tmp_path / "training"),
        options=dict(environment_steps=10000000, worlds=256),
    )


def test_training_rounds_down_and_selects_bounded_recipe(tmp_path):
    values = overrides(training_request(tmp_path))
    assert "training.total_environment_steps=9999872" in values
    assert "experiment=g1_bounded_flash_sac" in values
    assert "logging.mode=disabled" in values


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_training_budget(tmp_path, bad):
    request = training_request(tmp_path)
    request["options"]["environment_steps"] = bad
    with pytest.raises(ValueError, match="positive"):
        overrides(request)


def test_extra_overrides_cannot_escape_budget(tmp_path):
    request = training_request(tmp_path)
    request["options"]["overrides"] = ["training.total_environment_steps=1000000000"]
    with pytest.raises(ValueError, match="explicit stage options"):
        overrides(request)


def test_hydra_training_configuration():
    pytest.importorskip("hydra")
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.lifecycle.curriculum import FixedCurriculum

    c = compose_config("train_flash_sac", ["experiment=g1_bounded_flash_sac"])
    curriculum = instantiate_typed(c.curriculum, FixedCurriculum)
    assert c.action.hand_closing.closed_distance == 0.05
    assert curriculum.stages[-1].voc_scale == 0
    assert c.training.replay.capacity == 1000000
    assert c.logging.mode == "disabled"


def test_reference_assets_resolve_after_move(tmp_path):
    from flash_chord.assets.registry import resolve_asset_path

    moved = tmp_path / "moved"
    moved.mkdir()
    mesh = moved / "objects" / "mesh.obj"
    mesh.parent.mkdir()
    mesh.write_text("fixture")
    assert resolve_asset_path(
        "objects/mesh.obj", reference_path=moved / "reference.parquet"
    ) == str(mesh)
