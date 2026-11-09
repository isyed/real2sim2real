import numpy as np
import pytest

from real2sim2real.reconstruction_repair.elongated_shape import cross_section_audit


def case(crossbar):
    mask = np.zeros((200, 200), bool)
    mask[20:180, 96:104] = True
    mask[160:170, 55:145] = True
    depth = np.full(mask.shape, 2.)
    k = np.array([[400., 0., 100.], [0., 400., 100.], [0., 0., 1.]])
    z = np.linspace(0, .8, 100)
    shaft = np.column_stack((np.tile([-.01, .01], 50), np.zeros(100), z))
    if crossbar:
        broad = np.column_stack((np.linspace(-.45, .45, 100), np.zeros(100), np.full(100, .75)))
        shaft = np.vstack((shaft, broad))
    return cross_section_audit(mask, depth, k, shaft)


def test_missing_crossbar_is_flagged():
    result = case(False)
    assert result['review_required']
    assert result['observed_local_width_m'] > .3
    assert result['mesh_second_principal_extent_m'] < .1
    assert not result['mesh_modified']


def test_present_crossbar_is_not_flagged():
    result = case(True)
    assert not result['review_required']


def test_bad_depth_is_rejected():
    mask = np.ones((20, 20), bool)
    with pytest.raises(ValueError, match='depth'):
        cross_section_audit(mask, np.zeros((20, 20)), np.eye(3), np.ones((3, 3)))
