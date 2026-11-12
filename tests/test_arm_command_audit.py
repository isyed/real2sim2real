import numpy as np
import pytest
from real2sim2real.controller_actor_training.arm_command_audit import decompose


def test_plan_command_change_and_tracking_error_sum_in_a_shared_world():
    planned = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    commanded = planned + [0.002, -0.001, 0.003]
    actual = commanded + [-0.001, 0.004, -0.002]
    before = planned.copy()
    result = decompose(planned, commanded, actual)
    np.testing.assert_allclose(
        result["command_change_m"] + result["tracking_error_m"],
        result["total_error_m"],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result["tracking_error_m"], np.tile([-0.001, 0.004, -0.002], (2, 1))
    )
    translated = decompose(planned + 10, commanded + 10, actual + 10)
    np.testing.assert_allclose(
        translated["tracking_error_m"], result["tracking_error_m"], atol=1e-14
    )
    np.testing.assert_array_equal(planned, before)
    with pytest.raises(ValueError, match="Aligned"):
        decompose(planned, commanded[:1], actual)
