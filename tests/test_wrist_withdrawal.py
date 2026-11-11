import numpy as np
import pytest

from real2sim2real.controller_actor_training.wrist_withdrawal import (
    validate_withdrawal,
    offset,
    release_ready,
)


def settings():
    return dict(
        delay_s=0.5,
        duration_s=0.8,
        vectors_world_m=dict(right=[0.02, 0, 0.01], left=[-0.018, 0.009, 0.01]),
    )


def test_withdrawal_stays_on_release_clock_and_requires_supported_unhooking():
    s = settings()
    release = dict(start_s=13.0, duration_s=1.5)
    validate_withdrawal(s, release)
    np.testing.assert_array_equal(offset(13.5, "right", s, release), [0, 0, 0])
    np.testing.assert_allclose(offset(13.9, "right", s, release), [0.01, 0, 0.005])
    np.testing.assert_allclose(
        offset(20, "left", s, release), s["vectors_world_m"]["left"]
    )
    assert release_ready(13.4, s, release, None)
    assert not release_ready(13.5, s, release, None)
    assert not release_ready(13.5, s, release, 13.1)
    assert release_ready(13.5, s, release, 13.001)
    for v in [[0.04, 0, 0], [0, 0, -0.001], [float("nan"), 0, 0]]:
        bad = settings()
        bad["vectors_world_m"]["right"] = v
        with pytest.raises(ValueError):
            validate_withdrawal(bad, release)
    with pytest.raises(ValueError):
        validate_withdrawal(dict(s, delay_s=1.0), release)
    with pytest.raises(ValueError, match="thumb"):
        validate_withdrawal(s, dict(release, thumb_duration_s=0.75))
