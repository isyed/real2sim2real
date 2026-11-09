import numpy as np
import pytest

from real2sim2real.reconstruction_repair.static_floor_sequence import inspect_static_patches, fit_sequence


def test_camera_motion_and_texture_fail_closed():
    rng = np.random.default_rng(17)
    image = rng.normal(100, 20, (100, 180))
    rectangles = [[0, 0, 85, 100], [90, 0, 180, 100]]
    assert inspect_static_patches(image, image, rectangles)['passed']
    assert not inspect_static_patches(image, np.roll(image, 4, axis=1), rectangles)['passed']
    assert not inspect_static_patches(image*0+100, image*0+100, rectangles)['passed']


def test_sequence_keeps_original_anchor_and_reports_bad_frames():
    height, width = 90, 180
    K = np.array([[100., 0, 90], [0, 100, -60], [0, 0, 1]])
    v, u = np.indices((height, width))
    # Plane y=1.5 in CV coordinates, camera above floor.
    raw = 1.5/((v+60)/100)
    selection = dict(image_size=[width, height], rectangles_xyxy=[[0, 0, 85, 90], [90, 0, 180, 90]])
    empty = np.zeros(raw.shape, bool)
    def read(i):
        depth = raw.copy()
        if i == 1: depth *= 1.04
        if i == 2: depth[:, 90:] *= 1.2  # Regions disagree: never fill this gap.
        return depth, K, empty, empty
    result = fit_sequence(3, read, selection, [0., -1., 0., 1.5], 1.,
                          camera_evidence=dict(passed=True, source_sha256='synthetic'), pixel_stride=1)
    assert not result['complete']
    assert result['failed_frame_indices'] == [2]
    assert result['frames'][0]['gain'] == 1.
    assert result['frames'][1]['gain'] == pytest.approx(1/1.04)
    assert result['frames'][2]['gain'] is None
    assert not result['failed_frames_filled']


def test_sequence_requires_camera_evidence():
    with pytest.raises(ValueError):
        fit_sequence(2, None, {}, None, 1., camera_evidence=dict(passed=False))
