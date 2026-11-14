import pytest
from real2sim2real.controller_actor_training.contact_quality import contact_quality


def test_single_pad_loss_is_not_total_contact_loss():
    r = contact_quality(
        [[0, 2, 3], [2, 0, 3], [0, 0, 0]],
        [0.005, 0.02, 0.01],
        [20, 26, 25],
        minimum_force_n=0.1,
        maximum_drift_m=0.01,
        maximum_angle_deg=25,
    )
    assert r["any_pad_loss_fraction"] == 1
    assert r["all_pad_loss_fraction"] == pytest.approx(1 / 3)
    assert r["drift_exceedance_fraction"] == pytest.approx(1 / 3)
    assert not r["sampled_limits_pass"]


def test_missing_or_nonfinite_evidence_cannot_pass():
    for forces, drift, angle in [
        ([], [], []),
        ([[1, 2]], [float("nan")], [0]),
        ([[1, 2]], [], [0]),
    ]:
        with pytest.raises(ValueError):
            contact_quality(
                forces,
                drift,
                angle,
                minimum_force_n=0.1,
                maximum_drift_m=0.01,
                maximum_angle_deg=25,
            )
