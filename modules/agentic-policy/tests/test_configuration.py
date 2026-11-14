import json

import numpy as np
import pytest
from real2sim2real.agentic_policy import campaign
from real2sim2real.agentic_policy.authorization import scope
from real2sim2real.agentic_policy.diagnostics import retention
from real2sim2real.agentic_policy.objectives import assessment
from real2sim2real.agentic_policy.storage import load


def test_preview_does_not_call_author_or_consume_budget(tmp_path, monkeypatch):
    bundle = tmp_path / "input"
    (bundle / "author").mkdir(parents=True)
    (bundle / "experiment.json").write_text(
        json.dumps(
            {
                "mode": "automatic",
                "instruction": "test",
                "feedback_revisions": 6,
                "rollout_timeout_s": 5,
                "backend": {"kind": "diagnostic_counter"},
                "author": {"argv": ["never-execute"], "network": True},
            }
        )
    )
    root = tmp_path / "campaign"
    campaign.initialize(bundle, root)
    monkeypatch.setattr(
        campaign.sandbox,
        "run",
        lambda *a, **k: pytest.fail("Unexpected author execution"),
    )
    before = load(root)
    campaign.preview_author(root, tmp_path / "preview")
    assert load(root) == before
    assert (tmp_path / "preview/EVALUATION.json").exists()
    with pytest.raises(PermissionError, match="authorization"):
        campaign.advance(root)
    assert load(root)["author_attempts"] == []
    (root / "author-authorization.json").write_text(
        json.dumps(
            {
                "approved": True,
                "scope_sha256": "wrong",
                "user_decision_reference": "test-user",
            }
        )
    )
    with pytest.raises(PermissionError, match="authorization"):
        campaign.advance(root)
    assert len(scope(before)) == 64


def test_case_identity_infrastructure_and_external_prerequisites():
    protocol = {
        "target": "control",
        "robustness_cases": [{"name": "heavy"}],
        "admission": {
            "control": {
                "limits": {"drift": {"max": 0.01}},
                "prerequisites": ["scene_verified"],
            }
        },
        "evidence": {"scene_verified": False},
    }
    success = {"verdict": {"task_completed": True, "drift": 0.005}}
    receipt = {**success, "robustness_cases": {"wrong": success}}
    assert not assessment(receipt, protocol)["robustness_passed"]
    receipt["robustness_cases"] = {"heavy": success}
    a = assessment(receipt, protocol)
    assert a["admission"]["control"]["simulation_criteria_passed"]
    assert not a["admission"]["control"]["admitted"]
    assert (
        a["next_objective"]
        == "preserve_simulation_acceptance_pending_external_evidence"
    )
    receipt["infrastructure_error"] = "timeout"
    assert not assessment(receipt, protocol)["nominal_completed"]


def test_feedback_traces_only_use_the_previous_authored_trial(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "author").mkdir(parents=True)
    (bundle / "experiment.json").write_text(
        json.dumps(
            {
                "mode": "manual",
                "instruction": "test",
                "feedback_revisions": 6,
                "rollout_timeout_s": 5,
                "backend": {"kind": "diagnostic_counter"},
            }
        )
    )
    root = tmp_path / "campaign"
    campaign.initialize(bundle, root)
    state = load(root)
    candidate = root / "rounds/P0/candidate"
    candidate.mkdir(parents=True)
    (candidate / "workflow.json").write_text("{}")
    trial = root / "rounds/P0/trial-0/diagnostics"
    trial.mkdir(parents=True)
    rows = [
        {
            "time_s": i * 0.02,
            "penetration_m": 0.001 if i == 7 else 0,
            "pad_forces_n": [1, 1, 1],
        }
        for i in range(24)
    ]
    (trial / "observations.json").write_text(json.dumps(rows))
    (root / "manual-private.json").write_text("not an author input")
    state["rounds"] = [
        {
            "id": "P0",
            "trials": ["rounds/P0/trial-0"],
            "result": {"verdict": {"task_completed": False}},
        }
    ]
    work = tmp_path / "author-review"
    campaign.author_workspace(root, state, work)
    trace = json.loads((work / "feedback-traces.json").read_text())
    assert set(trace) == {"diagnostics/observations.json"}
    assert any(r["time_s"] == 0.14 for r in trace["diagnostics/observations.json"])
    assert not (work / "manual-private.json").exists()


def test_retention_uses_airborne_states_and_does_not_count_release_as_slip():
    def row(z, x=0, support=0, forces=(1, 1, 1)):
        return {
            "object_com_m": [x, 0, z],
            "support_fraction": support,
            "wrist_position_m": [0, 0, z],
            "wrist_rotation": np.eye(3).tolist(),
            "object_rotation": np.eye(3).tolist(),
            "pad_forces_n": forces,
            "pad_alignment_deg": [10, None, 15],
        }

    p = {"airborne_rise_m": 0.02, "minimum_pad_force_n": 0.1}
    r = retention(
        [row(0), row(0.1), row(0.1, 0.01, forces=(0, 0, 0)), row(0.1, 1, support=1)], p
    )
    assert r["carry_samples"] == 2
    assert r["carry_drift_m"] == pytest.approx(0.01)
    assert r["two_pad_contact_fraction"] == 0.5
    assert retention([row(0)], p)["carry_drift_m"] is None


def test_variant_changes_mass_inertia_and_partner_friction_together(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.agentic_policy.variants import create_case

    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/model.xml").write_text("""<mujoco><worldbody>
      <geom name="table" type="box" size="1 1 .1" friction=".9 .01 .001"/>
      <body name="object" pos="0 0 1"><freejoint/>
      <inertial pos="0 0 0" mass=".2" diaginertia=".01 .02 .03"/>
      <geom type="sphere" size=".1" friction=".6 .005 .0001"/>
      </body></worldbody></mujoco>""")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene/model.xml"))
    np.savez(tmp_path / "initial.npz", qpos=m.qpos0, qvel=np.zeros(m.nv))
    (tmp_path / "runtime.json").write_text(
        json.dumps(
            {
                "scene": "scene/model.xml",
                "initial": "initial.npz",
                "object_body": "object",
            }
        )
    )
    create_case(
        tmp_path,
        "light",
        mass_scale=0.8,
        sliding_friction_scale=0.75,
        translation_m=[0.005, 0, 0],
    )
    c = mujoco.MjModel.from_xml_path(str(tmp_path / "scene/case-light.xml"))
    assert c.body_mass[1] == pytest.approx(0.16)
    assert c.geom_friction[0, 0] == pytest.approx(0.675)
    assert c.geom_friction[1, 0] == pytest.approx(0.45)
    with np.load(tmp_path / "cases/light/initial.npz") as a:
        assert a["qpos"][0] == pytest.approx(0.005)
