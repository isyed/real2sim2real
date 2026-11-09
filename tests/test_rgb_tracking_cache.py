from pathlib import Path
import importlib
import numpy as np
import pytest


def test_written_cache_passes_the_actual_vendored_tracker_reader(tmp_path,monkeypatch):
    import h5py
    from real2sim2real.reconstruction.rgb_cache import write_frames
    root=Path(__file__).resolve().parents[1]/'modules/reconstruction/src/r2s2r_engines/cari4d/lib/cari4d'
    monkeypatch.syspath_prepend(str(root));reader=importlib.import_module('prep.mhr_rgb_h5')
    path=tmp_path/'rgb.h5';write_frames([np.full((12,16,3),i,np.uint8) for i in [10,20]],path,expected_count=2)
    with h5py.File(path) as f:
        assert reader.rgb_h5_frame_count(f)==2
        assert reader.validate_rgb_jpeg_metadata(f['frames'])==(12,16,3)
        assert reader.validate_rgb_jpeg(f['frames'][1],(12,16,3)).shape==(12,16,3)
    with h5py.File(path,'r+') as f:del f['frames'].attrs['optimize']
    with h5py.File(path) as f:
        with pytest.raises(ValueError,match='optimize'):reader.rgb_h5_frame_count(f)


def test_incomplete_or_retimed_stream_is_never_published_complete(tmp_path):
    import h5py
    from real2sim2real.reconstruction.rgb_cache import write_frames
    path=tmp_path/'short.h5'
    with pytest.raises(ValueError,match='ended'):write_frames([np.zeros((12,16,3),np.uint8)],path,expected_count=2)
    with h5py.File(path) as f:assert not f.attrs['complete']
    with pytest.raises(ValueError):write_frames([],path,expected_count=1)
