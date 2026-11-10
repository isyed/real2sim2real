from copy import deepcopy
import json

import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.grasp_sensitivity import (
    load_frozen_trial,
    validate_cases,
)


def cases():
    return [
        dict(name="nominal", mass_scale=1.0, sliding_friction=None),
        dict(name="heavy", mass_scale=1.25, sliding_friction=0.6),
    ]


def test_cases_require_nominal_and_reject_hidden_controller_or_unbounded_changes():
    validate_cases(cases())
    for changes in [
        dict(mass_scale=0),
        dict(sliding_friction=float("nan")),
        dict(name="../escape"),
        dict(gain=2),
        dict(name="nominal"),
    ]:
        bad = cases()
        bad[1].update(changes)
        with pytest.raises(ValueError):
            validate_cases(bad)
    bad = cases()
    bad[0]["sliding_friction"] = 0.8
    with pytest.raises(ValueError, match="nominal"):
        validate_cases(bad)


def frozen(tmp_path):
    directory = tmp_path / "trial"
    directory.mkdir()
    states = directory / "states.npz"
    states.write_bytes(b"immutable source states")
    result = dict(
        success=True,
        mass_scale=1.0,
        friction=0.8,
        simulation_final_time_s=10.0,
        simulation_requested_duration_s=10.0,
        states_sha256=digest(states),
        configuration=dict(physics=dict(friction=0.8)),
    )
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            dict(
                motion_condition="stationary_diagnostic",
                fidelity_limits=dict(position_m=0.02, rotation_deg=10),
            )
        )
    )
    config = dict(
        source_result=str(path),
        source_sha256=digest(path),
        protocol_sha256=digest(protocol),
        output=str(tmp_path / "output"),
        deadline_s=60,
        cases=cases(),
    )
    return config, result, path


def test_admission_rejects_incomplete_failed_changed_source_or_overrides(tmp_path):
    config, result, path = frozen(tmp_path)
    source, limits = load_frozen_trial(config)
    assert source == result and limits["position_m"] == 0.02
    for changes in [
        dict(success=False),
        dict(simulation_final_time_s=5),
        dict(mass_scale=1.25),
        dict(friction=0.6),
    ]:
        bad = dict(result, **changes)
        path.write_text(json.dumps(bad))
        pinned = dict(config, source_sha256=digest(path))
        with pytest.raises(ValueError, match="complete nominal"):
            load_frozen_trial(pinned)
    path.write_text(json.dumps(result))
    bad = dict(config, controller={"gain": 2})
    with pytest.raises(ValueError, match="overrides"):
        load_frozen_trial(bad)
    path.with_name("states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="states"):
        load_frozen_trial(config)


def test_replay_mismatch_stops_before_perturbations(tmp_path, monkeypatch):
    from real2sim2real.controller_actor_training import grasp_sensitivity as module
    from real2sim2real.controller_actor_training import (
        object_grasp_pilot,
        grasp_robustness,
        bilateral_evaluate,
    )

    config, source, _ = frozen(tmp_path)
    source["candidate"] = {"accepted": True}
    called = []
    monkeypatch.setattr(
        module, "load_frozen_trial", lambda c: (deepcopy(source), {"position_m": 0.02})
    )

    def trial(configuration, candidate, out, **kwargs):
        called.append(kwargs)
        out.mkdir()
        (out / "result.json").write_text("{}")
        return dict(source, states_sha256="different")

    monkeypatch.setattr(object_grasp_pilot, "trial", trial)
    monkeypatch.setattr(
        grasp_robustness, "summary", lambda r, d: dict(success=r["success"])
    )
    monkeypatch.setattr(
        bilateral_evaluate,
        "assess_motion_condition",
        lambda *a: dict(complete=True, diagnostic_hold_pass=True),
    )
    report = module.run(config)
    assert called == [dict(mass_scale=1.0, friction=None)]
    assert report["nominal_states_identical"] is False
    assert report["all_declared_cases_pass"] is False


def test_physical_pass_cannot_replace_demonstrated_reference_admission(
    tmp_path, monkeypatch
):
    from real2sim2real.controller_actor_training import retention_experiment

    config, source, path = frozen(tmp_path)
    source["configuration"]["reference_guidance"] = {"path": "independent-reference"}
    path.write_text(json.dumps(source))
    config["source_sha256"] = digest(path)
    protocol = path.parent.parent / "protocol.json"
    before = json.loads(protocol.read_text())
    before["motion_condition"] = "demonstrated"
    protocol.write_text(json.dumps(before))
    config["protocol_sha256"] = digest(protocol)
    monkeypatch.setattr(
        retention_experiment, "assess", lambda *args: dict(combined_pass=False)
    )
    with pytest.raises(ValueError, match="reference acceptance"):
        load_frozen_trial(config)
    monkeypatch.setattr(
        retention_experiment, "assess", lambda *args: dict(combined_pass=True)
    )
    assert load_frozen_trial(config)[0] == source


def test_reference_never_entered_is_a_complete_false_admission(monkeypatch):
    from real2sim2real.controller_actor_training import retention_experiment as module

    monkeypatch.setattr(module, "load_reference", lambda _: object())
    score = module.assess(
        dict(
            configuration=dict(reference_guidance={}),
            reference_start_time_s=None,
            simulation_final_time_s=1.0,
            simulation_requested_duration_s=10.0,
        ),
        {},
    )
    assert score == dict(
        combined_pass=False, reference_pass=False, fidelity=None, complete=False
    )


def test_review_inherits_only_pinned_source_acceptance(tmp_path):
    from real2sim2real.controller_actor_training.object_grasp_review import (
        protocol_fidelity_limits,
    )

    config, _, _ = frozen(tmp_path)
    protocol = tmp_path / "campaign-protocol.json"
    protocol.write_text(json.dumps(config))
    assert protocol_fidelity_limits(protocol)["position_m"] == 0.02
    source_protocol = tmp_path / "protocol.json"
    source_protocol.write_text(
        json.dumps(dict(fidelity_limits=dict(position_m=0.1, rotation_deg=30)))
    )
    with pytest.raises(ValueError, match="protocol changed"):
        protocol_fidelity_limits(protocol)
