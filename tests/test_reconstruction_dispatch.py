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
