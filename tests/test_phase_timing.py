import numpy as np
import pytest
from real2sim2real.controller_actor_training.phase_timing import PhaseTiming


def test_retiming_preserves_prefix_and_shifts_later_phases_without_pause():
    clock = PhaseTiming([0, 3, 5, 10], [0, 3, 7, 12])
    assert clock.reference(2.75) == 2.75
    assert clock.reference(5) == 4
    assert clock.reference(9) == 7
    assert clock.execution(10) == clock.duration_s == 12
    t = np.linspace(0, 12, 1001)
    assert (np.diff([clock.reference(x) for x in t]) > 0).all()
    for x in np.linspace(0, 10, 21):
        assert clock.reference(clock.execution(x)) == pytest.approx(x)
    with pytest.raises(ValueError):
        PhaseTiming([0, 3, 3, 10], [0, 3, 7, 12])
