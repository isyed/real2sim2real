import pytest
from real2sim2real.artifacts import digest, write_json
from real2sim2real.evidence_assessment import assess


def test_required_cohort_never_promotes_missing_or_rejected_claim(tmp_path):
    path = tmp_path / "evidence.json"
    write_json(path, {"nominal": True, "robust": False})
    config = {
        "scope": "Declared robustness cohort",
        "checks": [
            {"name": f, "report": "evidence.json", "sha256": digest(path), "field": f}
            for f in ["nominal", "robust", "unobserved"]
        ],
    }
    report = assess(config, tmp_path)
    assert not report["all_required_claims_pass"]
    assert [r["status"] for r in report["checks"]] == ["passed", "failed", "unresolved"]
    assert not report["automatic_training_or_hardware_promotion"]
    with pytest.raises(ValueError):
        assess(dict(config, checks=[]), tmp_path)
    with pytest.raises(ValueError):
        assess(dict(config, checks=config["checks"] * 2), tmp_path)
    write_json(path, {"nominal": True, "robust": True})
    with pytest.raises(ValueError, match="Evidence changed"):
        assess(config, tmp_path)


def test_truthy_values_are_not_boolean_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    write_json(path, {"accepted": "false"})
    with pytest.raises(ValueError, match="Boolean"):
        assess(
            {
                "scope": "test",
                "checks": [
                    {
                        "name": "test",
                        "report": "evidence.json",
                        "sha256": digest(path),
                        "field": "accepted",
                    }
                ],
            },
            tmp_path,
        )
