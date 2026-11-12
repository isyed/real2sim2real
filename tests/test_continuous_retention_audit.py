from copy import deepcopy

import pytest

from real2sim2real.controller_actor_training.continuous_retention_audit import (
    load_readiness,
)


def rows(times):
    labels = [
        role + ":" + group
        for role in ("primary", "secondary")
        for group in ("thumb", "index", "middle", "ring", "pinky")
    ]
    return [
        dict(
            time_s=t,
            contact_patch_motion=dict(pads={k: dict(force_n=2) for k in labels}),
            contact_force_feedback={k: dict(target_n=4) for k in labels},
        )
        for t in times
    ]


def test_sampled_load_readiness_needs_sustained_bilateral_opposition():
    data = rows([0, 0.02, 0.04, 0.06, 0.08, 0.10])
    original = deepcopy(data)
    report = load_readiness(data)
    assert report["first_confirmed_s"] == dict(
        primary=0.1, secondary=0.1, bilateral=0.1
    )
    assert data == original and report["acceptance_changed"] is False
    for row in data:
        for group in ("middle", "ring", "pinky"):
            row["contact_patch_motion"]["pads"]["primary:" + group]["force_n"] = 0
    assert load_readiness(data)["first_confirmed_s"]["bilateral"] is None


def test_gaps_and_brief_touch_do_not_create_a_load_certificate():
    data = rows([0, 0.02, 0.04, 0.08, 0.10, 0.12])
    assert load_readiness(data)["first_confirmed_s"]["bilateral"] is None
    data = rows([0, 0.02, 0.04, 0.06, 0.08, 0.10])
    data[3]["contact_patch_motion"]["pads"]["secondary:thumb"]["force_n"] = 0
    assert load_readiness(data)["first_confirmed_s"]["bilateral"] is None
    data[3]["time_s"] = 0.04
    with pytest.raises(ValueError, match="clock"):
        load_readiness(data)
