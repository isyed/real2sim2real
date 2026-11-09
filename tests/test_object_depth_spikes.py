import numpy as np
import pytest

from real2sim2real.reconstruction_repair.object_depth_spikes import plan_masked_depth_spikes


def test_bracketed_depth_spike_is_planned_without_changing_observations():
    depth = np.full(35, 3.2)
    depth[15:18] = [3.5, 4.1, 3.6]
    original = depth.copy()
    area = np.full(35, 1200)
    center = np.tile([500., 600.], (35, 1))
    plan = plan_masked_depth_spikes(depth, area, center)
    assert [row['frame'] for row in plan] == [15, 16, 17]
    assert all(abs(row['repaired_median_m'] - 3.2) < 1e-8 for row in plan)
    assert np.array_equal(depth, original)


def test_unbracketed_or_large_motion_is_not_silently_repaired():
    depth = np.full(35, 3.2)
    depth[15:18] = 4.1
    area = np.full(35, 1200)
    center = np.tile([500., 600.], (35, 1))
    center[16] = [900., 600.]
    plan = plan_masked_depth_spikes(depth, area, center)
    assert 16 not in [row['frame'] for row in plan]
    depth[0] = 4.0
    plan = plan_masked_depth_spikes(depth, area, center)
    assert 0 not in [row['frame'] for row in plan]


def test_rejects_incompatible_timelines():
    with pytest.raises(ValueError):
        plan_masked_depth_spikes(np.ones(10), np.ones(9), np.ones((10, 2)))
