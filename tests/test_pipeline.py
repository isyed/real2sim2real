import json
from pathlib import Path
import sys

import pytest

from real2sim2real.artifacts import digest, load, publish, write_json
from real2sim2real.pipeline import STAGES, execute, plan


def config(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"synthetic input")
    return dict(
        schema="real2sim2real.pipeline/1",
        input="video.mp4",
        output="output",
        stages=["reconstruction"],
        modules={"reconstruction": {}},
    )


def test_stage_selection_and_type_errors(tmp_path):
    c = config(tmp_path)
    assert plan(c, tmp_path)[0]["output_kind"] == "reconstruction"
    c["modules"]["retargeting"] = {}
    with pytest.raises(ValueError, match="received video"):
        plan(c, tmp_path, selected=["retargeting"])
    c["input_kind"] = "reconstruction"
    assert (
        plan(c, tmp_path, selected=["retargeting"])[0]["output_kind"]
        == "robot-reference"
    )
    with pytest.raises(ValueError, match="duplicates"):
        plan(c, tmp_path, selected=["retargeting", "retargeting"])


def test_reject_overwrite_and_invalid_deadline(tmp_path):
    c = config(tmp_path)
    c["output"] = c["input"]
    with pytest.raises(ValueError, match="separate"):
        plan(c, tmp_path)
    c["output"] = "fresh"
    c["modules"]["reconstruction"]["execution"] = {"timeout_seconds": -1}
    with pytest.raises(ValueError, match="positive"):
        plan(c, tmp_path)


def test_relative_interpreter(tmp_path):
    c = config(tmp_path)
    c["modules"]["reconstruction"]["execution"] = {"python": "env/bin/python"}
    assert plan(c, tmp_path)[0]["python"] == str(tmp_path / "env/bin/python")


def test_content_hash_and_path_boundaries(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    data = root / "data.txt"
    data.write_text("original")
    publish(root, "test", {"data": data})
    assert load(root)["kind"] == "test"
    data.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        load(root)
    value = json.loads((root / "artifact.json").read_text())
    value["files"]["data"]["path"] = "../outside"
    write_json(root / "artifact.json", value)
    with pytest.raises(ValueError, match="escapes"):
        load(root)


def test_publish_rejects_symlink(tmp_path):
    data = tmp_path / "data"
    data.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(data)
    with pytest.raises(ValueError, match="symlink"):
        publish(tmp_path, "test", {"bad": link})


def test_real_process_chain_and_resume_tamper(tmp_path, monkeypatch):
    # The fixture is deliberately not a model: it exercises the actual process,
    # artifact, receipt and resume protocol with no GPU or external asset.
    (tmp_path / "fixture_stage.py").write_text("""
from pathlib import Path
from real2sim2real.artifacts import publish
from real2sim2real.stage import request_main
def run(request):
    root=Path(request['output']); p=root/'result.txt'; p.write_text('computed')
    publish(root,request['output_kind'],{'result':p})
if __name__=='__main__': request_main(run)
""")
    import os

    search = [str(tmp_path)] + [
        str(Path(p).resolve())
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p
    ]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(search))
    monkeypatch.setitem(
        STAGES, "fixture", ("fixture_stage", {"video"}, "fixture-output")
    )
    c = config(tmp_path)
    c.update(stages=["fixture"], modules={"fixture": {}})
    path = tmp_path / "pipeline.json"
    write_json(path, c)
    output = execute(path)
    assert load(output)["kind"] == "fixture-output"
    assert execute(path, resume=True) == output
    value = json.loads((output / "artifact.json").read_text())
    value["metadata"]["unrecorded_change"] = True
    write_json(output / "artifact.json", value)
    with pytest.raises(ValueError, match="manifest changed"):
        execute(path, resume=True)


def test_thermal_settings_rejected_before_launch(tmp_path):
    from real2sim2real.supervisor import run

    with pytest.raises(ValueError, match="Thermal"):
        run(
            ["does-not-exist"],
            cwd=tmp_path,
            env={},
            stream=sys.stdout,
            timeout=1,
            thermal=dict(pause_c=74, resume_c=82, stop_c=83),
        )


def test_timeout_stops_actual_process(tmp_path):
    from real2sim2real.supervisor import run

    with (tmp_path / "log").open("w") as log, pytest.raises(TimeoutError):
        run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env={},
            stream=log,
            timeout=0.05,
        )
