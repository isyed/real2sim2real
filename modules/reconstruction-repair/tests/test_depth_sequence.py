import io

import h5py
import numpy as np
from PIL import Image
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.reconstruction_repair.depth_sequence import materialize


def archive(path):
    camera = 'front_stereo_camera_left'
    raw = np.full((20, 24), 2000, np.uint16)
    raw[0, :2] = [0, 65535]
    buffer = io.BytesIO(); Image.fromarray(raw).save(buffer, format='PNG',compress_level=3,optimize=False)
    payload = np.frombuffer(buffer.getvalue(),np.uint8)
    with h5py.File(path, 'w') as f:
        f.attrs.update(format='cari4d_mhr_metric_depth_png_v1', complete=True)
        for group in ['raw', 'aligned']:
            d = f.create_dataset(group+'/'+camera, (3,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for i in range(3): d[i] = payload
        f.create_dataset('frame_names/'+camera, data=['0','1','2'], dtype=h5py.string_dtype())
        for key, data in [('scale',np.ones(3)), ('shift',np.zeros(3)), ('valid_count',np.full(3,478))]:
            f.create_dataset('alignment/'+camera+'/'+key,data=data)
    return camera, payload


def test_materialized_depth_keeps_anchor_and_raw_payload(tmp_path):
    source = tmp_path/'source.h5'; camera, payload = archive(source)
    source_hash = digest(source)
    out = tmp_path/'new.h5'
    report = materialize(source, [1.,1.05,.95], out, source_sha256=source_hash,evidence_sha256='synthetic')
    with h5py.File(out) as f:
        assert f.attrs['complete']
        assert np.array_equal(f['raw/'+camera][2],payload)
        assert np.array_equal(f['aligned/'+camera][0],payload)
        z = np.asarray(Image.open(io.BytesIO(f['aligned/'+camera][1].tobytes())))
        assert z[1,1] == 2100
        assert z[0,0] == 0 and z[0,1] == 65535
    assert digest(source) == source_hash
    assert report['requires_fresh_tracking'] and not report['old_poses_approved']


@pytest.mark.parametrize('gains', [[1.,1.2,1.], [1.,float('nan'),1.], [1.,1.], [1.01,1.,1.]])
def test_bad_gain_or_timeline_rejected(tmp_path,gains):
    source=tmp_path/'source.h5';archive(source)
    with pytest.raises(ValueError):
        materialize(source,gains,tmp_path/'bad.h5',source_sha256=digest(source),evidence_sha256='synthetic')


def test_source_hash_and_existing_output_rejected(tmp_path):
    source=tmp_path/'source.h5';archive(source)
    with pytest.raises(ValueError):
        materialize(source,[1.,1.,1.],tmp_path/'out.h5',source_sha256='wrong',evidence_sha256='synthetic')
    output=tmp_path/'exists.h5';output.touch()
    with pytest.raises(FileExistsError):
        materialize(source,[1.,1.,1.],output,source_sha256=digest(source),evidence_sha256='synthetic')
