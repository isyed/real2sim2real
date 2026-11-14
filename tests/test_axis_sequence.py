import numpy as np
from real2sim2real.egocentric.axis_sequence import fit_axis_sequence
from real2sim2real.egocentric.container_axis import endpoints


def test_sequence_recovers_smooth_observed_translation_without_changing_clock():
    times = np.array([2.0, 2.05, 2.11, 2.16, 2.22])
    dims = np.array([0.03, 0.04, 0.15])
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    true = np.repeat(np.eye(4)[None], len(times), axis=0)
    true[:, 2, 3] = 0.7
    true[:, 0, 3] = (times - times[0]) * 0.05
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    z = np.tile([0.02, 0.06, 0.10, 0.14], 6)
    r = dims[0] + (dims[1] - dims[0]) * z / dims[2]
    local = np.c_[r * np.cos(th), r * np.sin(th), z]
    points = [local + T[:3, 3] for T in true]
    cues = [endpoints(T, dims[2], K) for T in true]
    seed = true.copy()
    seed[:, 0, 3] += [0.003, -0.003, 0.004, -0.003, 0.003]
    fitted, report = fit_axis_sequence(
        points,
        cues,
        np.repeat(np.eye(4)[None], len(times), axis=0),
        K,
        dims,
        times,
        seed,
        max_evaluations=60,
    )
    assert np.max(np.linalg.norm(fitted[:, :3, 3] - true[:, :3, 3], axis=1)) < 0.001
    assert report["time_warped"] is False and report["axial_spin_observed"] is False
