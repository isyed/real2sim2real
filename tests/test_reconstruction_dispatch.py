"""Check the environment boundary without loading model libraries."""

import json
from pathlib import Path
import subprocess
import sys
import pytest
from real2sim2real.artifacts import digest
from real2sim2real.reconstruction import stage


def request(tmp_path):
    video = tmp_path / "input with spaces.mp4"
    video.write_bytes(b"fixture")
    return dict(
        input=str(video),
        input_sha256=digest(video),
        output=str(tmp_path / "output"),
        config_base=str(tmp_path),
        options={},
    )


def test_container_prefix_preserves_arguments(tmp_path):
    r = request(tmp_path)
    r["options"]["backend_commands"] = {
        "metric_mesh": ["wrapper", "--label", "a $literal value", "--", "python"]
    }
    cmd = stage.command(r, "metric_mesh", tmp_path / "request with spaces.json")
    assert cmd[:5] == ["wrapper", "--label", "a $literal value", "--", "python"]
    assert cmd[-1] == str(tmp_path / "request with spaces.json")
    assert "--phase" in cmd


@pytest.mark.parametrize("bad", ["python -c anything", [], [""], [3]])
def test_reject_shell_strings_and_empty_prefixes(tmp_path, bad):
    r = request(tmp_path)
    r["options"]["backend_commands"] = {"video": bad}
    with pytest.raises(ValueError, match="argument lists"):
        stage.run(r)
    assert not (Path(r["output"]) / "backend_request.json").exists()


def test_failed_phase_stops_chain_and_records_failure(tmp_path, monkeypatch):
    r = request(tmp_path)
    calls = []

    def fake(argv, **kwargs):
        phase = argv[argv.index("--phase") + 1]
        calls.append(phase)
        if phase == "depth":
            raise subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(stage.subprocess, "run", fake)
    with pytest.raises(subprocess.CalledProcessError):
        stage.run(r)
    assert calls == ["video", "segment", "pack_masks", "depth"]
    status = json.loads((Path(r["output"]) / "phase_status.json").read_text())
    assert status["state"] == "failed" and status["phase"] == "depth"
    with pytest.raises(FileExistsError):
        stage.run(r)


def test_input_mutation_rejected_before_any_model(tmp_path):
    r = request(tmp_path)
    Path(r["input"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source video changed"):
        stage.run(r)
    assert not Path(r["output"]).exists()


def test_dispatcher_imports_without_scientific_dependencies():
    code = 'import sys; from real2sim2real.reconstruction import stage; assert not ({"torch","numpy","av","trimesh"} & sys.modules.keys())'
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("phase", sorted(stage.GPU_PHASES))
def test_full_backend_check_hides_cuda_and_only_requests_help(tmp_path, monkeypatch, phase):
    from real2sim2real.reconstruction import backend

    r = request(tmp_path)
    weights = tmp_path / "weights"
    for name in [
        "sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt",
        "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    ]:
        p = weights / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not loaded")
    prompts = tmp_path / "prompts.json"
    prompts.write_text("{}")
    r["options"].update(
        weights=str(weights),
        sam2_weights=str(weights),
        sam3d_weights=str(weights),
        prompts=str(prompts),
    )
    monkeypatch.setattr(backend, "engine_paths", lambda: (tmp_path, tmp_path, tmp_path))
    monkeypatch.setattr(backend.os, "environ", {"CUDA_VISIBLE_DEVICES": "0"})
    captured = []
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda args, **kwargs: captured.append((args, kwargs)),
    )
    backend.run(r, phase, check_backend=True)
    assert len(captured) == 1
    argv, kwargs = captured[0]
    assert argv[-1] == "--help" and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not Path(r["output"]).exists()
    with pytest.raises(ValueError, match="model phases"):
        backend.run(r, "publish", check_backend=True)


def test_object_and_human_model_caches_are_separate(tmp_path):
    from real2sim2real.reconstruction.backend import cache_environment

    human = tmp_path / "human-models"
    obj = tmp_path / "object-models"
    for phase in stage.GPU_PHASES:
        env = cache_environment(phase, human, obj)
        if phase == "metric_mesh":
            assert Path(env["TORCH_HOME"]) == obj / "torch_home"
            assert Path(env["HF_HOME"]) == obj / "hf_home"
        else:
            assert Path(env["TORCH_HOME"]) == human / "sam3d_body/torch_home"
            assert Path(env["HF_HOME"]) == human / "hf_home"
