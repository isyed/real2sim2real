import pytest
from real2sim2real.agentic_policy.transport_diagnostics import rejection_evidence


def test_contact_aware_blocker_supersedes_larger_exempt_pad_overlap():
    plan = {
        "admitted": False,
        "screen": {
            "admitted": False,
            "peak_penetration_m": 0.0042,
            "witness": ["cup", "pad"],
        },
        "carried_object_screen": {"admitted": True, "peak_penetration_m": 0.0029},
        "contact_aware_screen": {
            "admitted": False,
            "peak_penetration_m": 0.0032,
            "witness": ["cup", "rigid_link"],
        },
    }
    result = rejection_evidence(plan)
    assert result["blockers"][0]["witness"] == ["cup", "rigid_link"]
    assert result["blockers"][0]["peak_penetration_m"] == 0.0032
    assert not result["live_collision_claim"]


def test_failed_loaded_forecast_is_reported_as_the_actual_decision():
    result = rejection_evidence(
        {
            "admitted": False,
            "contact_aware_screen": {
                "admitted": False,
                "peak_penetration_m": 0.004,
                "dynamic_preview": {
                    "passed": False,
                    "reason": "transport_contact_loss",
                },
            },
        }
    )
    assert result["blockers"][0]["source"].endswith("dynamic_preview")
    assert result["blockers"][0]["reason"] == "transport_contact_loss"


def test_legacy_screens_keep_both_rejections_and_ik_keeps_its_reason():
    result = rejection_evidence(
        {
            "admitted": False,
            "screen": {"admitted": False},
            "carried_object_screen": {"admitted": False},
        }
    )
    assert len(result["blockers"]) == 2
    assert (
        rejection_evidence({"admitted": False, "reason": "wrist IK rejected"})[
            "blockers"
        ][0]["reason"]
        == "wrist IK rejected"
    )
    with pytest.raises(ValueError):
        rejection_evidence({"admitted": True})
    with pytest.raises(ValueError):
        rejection_evidence(
            {"admitted": False, "contact_aware_screen": {"admitted": True}}
        )
