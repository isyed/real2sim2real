import numpy as np
import pytest

from real2sim2real.egocentric.demonstration_strategy import motion_rows


def test_motion_clearance_preserves_original_start_and_does_not_infer_contact():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 2, 3] = [0.02, 0.025, 0.07]
    vertices = np.array([[0, 0, 0], [0.1, 0, 0.01]])
    rows = motion_rows(poses, vertices, [3, 3.05, 3.1], [100, 101, 102])
    assert rows[0]["minimum_clearance_m"] == 0.02
    assert rows[-1]["translation_from_start_m"] == pytest.approx(0.05)
    assert [r["frame_id"] for r in rows] == [100, 101, 102]
    assert not any(r["contact_observed"] for r in rows)
    with pytest.raises(ValueError):
        motion_rows(poses, vertices, [3, 3.05, 3.05], [100, 101, 102])
