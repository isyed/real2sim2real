import h5py
import numpy as np
import pytest

from real2sim2real.reconstruction.tracking_inputs import slice_h5, window_indices


@pytest.mark.parametrize('indices', [[2, 3, 4], [4, 3, 2], [3]])
def test_window_indices_preserve_explicit_direction(indices):
    assert window_indices(indices, 5).tolist() == indices


@pytest.mark.parametrize('indices', [[], [1, 3], [2, 2], [-1, 0], [4, 5], [1.0, 2.0], [True]])
def test_window_rejects_ambiguous_or_invalid_source_indices(indices):
    with pytest.raises(ValueError):
        window_indices(indices, 5)


def test_hdf5_window_keeps_masks_depth_and_encoded_images_in_source_order(tmp_path):
    source, target = tmp_path/'source.h5', tmp_path/'window.h5'
    with h5py.File(source, 'w') as h:
        h.attrs['complete'] = True
        h.create_dataset('frames', data=np.arange(20).reshape(5, 2, 2))
        h.create_dataset('alignment/camera/scale', data=np.arange(5) + .5)
        names = h.create_dataset('frame_names/camera', shape=(5,), dtype=h5py.string_dtype())
        encoded = h.create_dataset('raw/camera', shape=(5,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(5):
            names[i] = f'{i:06d}'
            encoded[i] = np.array([i, i+1], np.uint8)
        encoded.attrs['codec'] = 'test'
    slice_h5(source, target, window_indices([4, 3, 2], 5), 5)
    with h5py.File(target) as h:
        assert h.attrs['complete']
        np.testing.assert_array_equal(h['frames'][:, 0, 0], [16, 12, 8])
        np.testing.assert_array_equal(h['alignment/camera/scale'][:], [4.5, 3.5, 2.5])
        assert list(h['frame_names/camera'].asstr()[:]) == ['000000', '000001', '000002']
        assert h['raw/camera'].attrs['codec'] == 'test'
        assert h['raw/camera'][0].tolist() == [4, 5]


def test_hdf5_window_refuses_unmapped_datasets_or_incomplete_sources(tmp_path):
    source = tmp_path/'source.h5'
    with h5py.File(source, 'w') as h:
        h.attrs['complete'] = False
        h.create_dataset('frames', data=np.arange(5))
    with pytest.raises(ValueError, match='complete'):
        slice_h5(source, tmp_path/'a.h5', [2, 3], 5)
    with h5py.File(source, 'a') as h:
        h.attrs['complete'] = True
        h.create_dataset('unmapped', data=np.arange(2))
    with pytest.raises(ValueError, match='Unmapped'):
        slice_h5(source, tmp_path/'b.h5', [2, 3], 5)
