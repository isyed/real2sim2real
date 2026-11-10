import pytest
from real2sim2real.controller_actor_training.object_control_stage import (
    bind_config,
    run,
)

from real2sim2real.artifacts import digest, load, publish, write_json
from real2sim2real.control_artifacts import adopt
from real2sim2real.pipeline import plan


def test_named_recipes_remain_composable_and_check_types(tmp_path):
    c = dict(
        pipeline="object-reference-control",
        input="grasp",
        output="out",
        modules={"arm-path-audit": {}, "arm-replay": {}},
    )
    rows = plan(c, tmp_path)
    assert [r["input_kind"] for r in rows] == ["grasp-trial", "arm-path-audit"]
    assert rows[-1]["output_kind"] == "object-control-evaluation"
    with pytest.raises(ValueError, match="received video"):
        plan(dict(c, input_kind="video"), tmp_path)
    assert (
        len(
            plan(
                dict(c, input_kind="arm-path-audit"), tmp_path, selected=["arm-replay"]
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="Unknown pipeline"):
        plan(dict(c, pipeline="typo"), tmp_path)


def grasp(tmp_path, success=True):
    src = tmp_path / "original"
    src.mkdir()
    (src / "states.npz").write_bytes(b"archived states")
    write_json(
        src / "result.json",
        dict(
            schema="real2sim2real.object-grasp-trial/2",
            success=success,
            states_sha256=digest(src / "states.npz"),
        ),
    )
    return src


def request(tmp_path, stage, inp, config):
    path = tmp_path / "config.json"
    write_json(path, config)
    out = tmp_path / "output"
    out.mkdir()
    return dict(
        stage=stage,
        input=str(inp),
        output=str(out),
        output_kind="arm-path-audit"
        if stage == "arm-path-audit"
        else "object-control-evaluation",
        config_base=str(tmp_path),
        options=dict(config="config.json", config_sha256=digest(path)),
    )


def test_adoption_preserves_sources_and_rejects_changed_states(tmp_path):
    src = grasp(tmp_path)
    original = digest(src / "result.json")
    dst = adopt(src, tmp_path / "adopted", "grasp-trial")
    assert digest(src / "result.json") == digest(dst / "result.json") == original
    assert load(dst)["metadata"]["training_ready_established"] is False
    (src / "states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        adopt(src, tmp_path / "bad", "grasp-trial")


def test_adapter_rebinds_upstream_and_pins_configuration(tmp_path):
    src = grasp(tmp_path)
    dst = adopt(src, tmp_path / "adopted", "grasp-trial")
    req = request(
        tmp_path,
        "arm-path-audit",
        dst,
        dict(
            trial="stale",
            sources={"stale": "sha", "other": "preserved"},
            output="stale-output",
        ),
    )
    cfg = bind_config(req)
    assert cfg["trial"] == str(dst / "result.json") and "stale" not in cfg["sources"]
    assert cfg["sources"][cfg["trial"]] == digest(dst / "result.json")
    assert cfg["sources"]["other"] == "preserved"
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="config_sha256"):
        bind_config(req)


def test_rejected_grasp_cannot_enter_arm_stage(tmp_path):
    src = grasp(tmp_path, False)
    dst = adopt(src, tmp_path / "adopted", "grasp-trial")
    req = request(tmp_path, "arm-path-audit", dst, dict(trial="stale", sources={}))
    with pytest.raises(ValueError, match="accepted grasp"):
        bind_config(req)


def test_failed_selected_path_never_invokes_physics(tmp_path, monkeypatch):
    from real2sim2real.controller_actor_training import arm_replay_pilot

    inp = tmp_path / "audit"
    inp.mkdir()
    write_json(
        inp / "audit.json",
        dict(
            placements=[
                dict(placement={"name": "chosen"}, screen_pass=False),
                dict(placement={"name": "other"}, screen_pass=True),
            ]
        ),
    )
    publish(inp, "arm-path-audit", {"audit": inp / "audit.json"})
    req = request(tmp_path, "arm-replay", inp, dict(placement="chosen"))

    def forbidden(config):
        pytest.fail("Physics must not run after rejection")

    monkeypatch.setattr(arm_replay_pilot, "run", forbidden)
    with pytest.raises(ValueError, match="complete arm path"):
        run(req)


def test_audit_stage_publishes_diagnostic_failure_without_promoting(
    tmp_path, monkeypatch
):
    from real2sim2real.controller_actor_training import arm_path_audit

    src = grasp(tmp_path)
    dst = adopt(src, tmp_path / "adopted", "grasp-trial")
    req = request(tmp_path, "arm-path-audit", dst, dict(trial="old", sources={}))

    def audit(config):
        from pathlib import Path

        out = Path(config["output"])
        out.mkdir()
        write_json(out / "audit.json", dict(placements=[dict(screen_pass=False)]))
        write_json(out / "assembly.json", {})
        (out / "audit-scene.xml").write_text("<mujoco/>")

    monkeypatch.setattr(arm_path_audit, "run", audit)
    run(req)
    result = load(req["output"])
    assert result["metadata"]["stage_accepted"] is False
    assert result["metadata"]["training_ready_established"] is False


def test_stage_cannot_bypass_shorter_module_deadline(tmp_path):
    src = grasp(tmp_path)
    dst = adopt(src, tmp_path / "adopted", "grasp-trial")
    req = request(
        tmp_path, "arm-path-audit", dst, dict(trial="stale", sources={}, deadline_s=60)
    )
    req["timeout_seconds"] = 900
    with pytest.raises(ValueError, match="module deadline"):
        bind_config(req)
    req["timeout_seconds"] = 60
    assert bind_config(req)["deadline_s"] == 60
