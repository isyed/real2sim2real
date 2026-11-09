import json

import pytest

from real2sim2real.egocentric.reuse import reuse_mesh, fork_tracking_inputs


def observations(root, *, count=2):
    root.mkdir()
    (root / "timeline.json").write_text(
        json.dumps(
            dict(
                capture_sha256="same-capture",
                intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                image_size=[8, 8],
                frames=count,
            )
        )
    )
    for name in ("rgb/000000.png", "depth/000000.npy", "masks/1/000000.png"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


def source_mesh(root):
    (root / "mesh").mkdir()
    (root / "mesh/metric.glb").write_bytes(b"mesh")
    (root / "mesh/input_quality.json").write_text('{"frame":0}')


def test_extension_reuses_only_identical_conditioning_inputs(tmp_path):
    prefix, full = tmp_path / "prefix", tmp_path / "full"
    observations(prefix)
    observations(full, count=20)
    source_mesh(prefix)
    reuse_mesh(prefix, full)
    assert (full / "mesh/metric.glb").read_bytes() == b"mesh"
    receipt = json.loads((full / "mesh/reuse.json").read_text())
    assert len(receipt["inputs"]) == 3 and "metric.glb" in receipt["files"]
    with pytest.raises(FileExistsError):
        reuse_mesh(prefix, full)


@pytest.mark.parametrize(
    "changed", ["rgb/000000.png", "depth/000000.npy", "masks/1/000000.png"]
)
def test_changed_observation_cannot_reuse_mesh(tmp_path, changed):
    prefix, full = tmp_path / "prefix", tmp_path / "full"
    observations(prefix)
    observations(full)
    source_mesh(prefix)
    (full / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        reuse_mesh(prefix, full)
    assert not (full / "mesh").exists()


@pytest.mark.parametrize(
    "phase, sentinel",
    [
        ("calibrate", "calibration.json"),
        ("prepare", "rgb"),
        ("camera", "camera_motion.npz"),
        ("publish", "artifact.json"),
    ],
)
def test_phase_retries_preserve_existing_results(tmp_path, phase, sentinel):
    from real2sim2real.egocentric.cli import execute

    path = tmp_path / sentinel
    path.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        execute({"output": str(tmp_path)}, phase)
    assert path.read_bytes() == b"previous result"


def test_tracking_fork_excludes_predictions_and_does_not_share_writable_files(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    observations(source)
    source_mesh(source)
    for name in (
        "camera.json",
        "camera_motion.npz",
        "camera_motion_quality.json",
        "calibration.json",
        "prompts.json",
        "depth_quality.json",
        "object_motion.npz",
    ):
        (source / name).write_bytes(b"original")
    fork_tracking_inputs(source, target)
    assert not (target / "object_motion.npz").exists()
    (target / "mesh/metric.glb").write_bytes(b"new hypothesis")
    assert (source / "mesh/metric.glb").read_bytes() == b"mesh"
    receipt = json.loads((target / "tracking_inputs.json").read_text())
    assert "mesh/metric.glb" in receipt["files"]


def test_tracking_fork_checks_required_inputs_before_copying(tmp_path):
    source = tmp_path / "source"
    observations(source)
    with pytest.raises(FileNotFoundError):
        fork_tracking_inputs(source, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_tracking_fork_rejects_recursive_copy_destination(tmp_path):
    source = tmp_path / "source"
    observations(source)
    with pytest.raises(ValueError, match="overlap"):
        fork_tracking_inputs(source, source / "rgb" / "nested")
