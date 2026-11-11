from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.finger_acquisition import (
    active,
    seeking,
    validate,
)
from real2sim2real.controller_actor_training.arm_finger_feedback import (
    feedback_settings,
)


def recipe():
    return dict(
        start_fraction=0.0,
        end_after_close_s=0.6,
        gain_m_per_n_s=0.003,
        max_normal_speed_m_s=0.006,
    )


def test_seeking_clock_preserves_original_acquisition_end_and_is_bounded():
    r, c = dict(approach_s=1.0, close_s=0.8), recipe()
    before = deepcopy(r)
    validate(c, 0.75)
    assert not active(0.999, r, c)
    assert active(1, r, c) and active(2.399, r, c)
    assert not active(2.4, r, c)
    assert r == before


def test_first_contact_latches_and_never_rearms_after_contact_loss():
    p = dict(group="index")
    assert seeking(p, 0, active=True, threshold_n=0.1)
    assert not seeking(p, 0.1, active=True, threshold_n=0.1)
    assert not seeking(p, 0, active=True, threshold_n=0.1)
    assert p["acquisition_contact_seen"] is True
    assert not seeking(dict(group="thumb"), 0, active=True, threshold_n=0.1)
    p = dict(group="ring")
    assert not seeking(p, 1, active=False, threshold_n=0.1)
    assert not seeking(p, 0, active=True, threshold_n=0.1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("start_fraction", 0.76),
        ("end_after_close_s", 1.01),
        ("gain_m_per_n_s", 0.0031),
        ("max_normal_speed_m_s", 0.0061),
        ("start_fraction", np.nan),
        ("end_after_close_s", True),
    ],
)
def test_invalid_acquisition_cannot_relax_clock_or_rate_caps(key, value):
    c = recipe()
    c[key] = value
    with pytest.raises(ValueError):
        validate(c, 0.75)


def test_acquisition_override_preserves_force_and_joint_authority():
    from test_contact_force_control import settings

    original = settings()
    override = dict(unloaded_acquisition=recipe())
    updated = feedback_settings(original, override)
    assert original == settings()
    assert {k: v for k, v in updated.items() if k != "unloaded_acquisition"} == original
    override["unloaded_acquisition"]["gain_m_per_n_s"] = 0.5
    assert updated["unloaded_acquisition"] == recipe()


def test_moving_acquisition_cannot_bypass_complete_stationary_admission(monkeypatch):
    import sys
    from types import SimpleNamespace
    from real2sim2real.controller_actor_training.arm_replay_pilot import rollout

    # Admission must reject before it accesses a model, audit or dynamics.
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace())
    with pytest.raises(ValueError, match="complete passing stationary"):
        next(
            rollout(
                dict(
                    finger_control_mode="live_contact_force",
                    finger_feedback_override=dict(unloaded_acquisition=recipe()),
                )
            )
        )
