"""Materialize bounded depth evidence using the reconstruction engine's codec.

This adapter composes reconstruction and repair explicitly. It does not alter
meshes or motion, and producing depth does not approve old poses for that depth.
"""
from pathlib import Path
import io

import numpy as np


def materialize(source, gains, output, *, source_sha256, evidence_sha256,
                camera='front_stereo_camera_left', progress=None):
    import h5py
    from PIL import Image
    from real2sim2real.artifacts import digest, write_json
    from r2s2r_engines.cari4d.lib.cari4d.prep.mhr_depth_h5 import MHRDepthH5Writer, DepthFrameRecord

    source, output = Path(source), Path(output)
    gains = np.asarray(gains, float)
    if (gains.ndim != 1 or len(gains) < 2 or not np.isfinite(gains).all()
            or np.any(gains <= 0) or not evidence_sha256
            or digest(source) != source_sha256):
        raise ValueError('Complete positive gains and verified source/evidence identities required')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Preserve previous materialized depth')
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(source_depth_sha256=source_sha256, evidence_sha256=evidence_sha256,
                    source_frame_zero_preserved=True, poses_revalidated=False)
    with h5py.File(source, 'r') as old:
        if (old.attrs.get('format') != 'cari4d_mhr_metric_depth_png_v1'
                or not old.attrs.get('complete') or len(old['raw/'+camera]) != len(gains)
                or len(old['aligned/'+camera]) != len(gains)
                or len(old['frame_names/'+camera]) != len(gains)):
            raise ValueError('Explicit complete native metric-PNG depth archive required')
        first_gain = float(old['alignment/'+camera+'/scale'][0])
        if gains[0] != first_gain or np.any((gains/first_gain < .9) | (gains/first_gain > 1.1)):
            raise ValueError('Original anchor or bounded gain contract differs')
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in old['frame_names/'+camera][:]]
        writer = MHRDepthH5Writer(output, {camera:names},
            alignment_method='predicted_static_floor_sequence_v1',
            alignment_input_identity=identity, encoding_workers=2, max_pending_batches=2)
        try:
            for i, gain in enumerate(gains):
                payload = old['raw/'+camera][i]
                millimetres = np.asarray(Image.open(io.BytesIO(payload.tobytes())))
                if millimetres.dtype.kind != 'u' or millimetres.ndim != 2:
                    raise ValueError('Raw depth must be native uint16 millimetres')
                raw = millimetres.astype(np.float32)/1000
                aligned = raw*gain
                invalid = (millimetres == 0) | (millimetres == 65535)
                aligned[invalid] = raw[invalid]
                if i == 0:
                    aligned = np.asarray(Image.open(io.BytesIO(old['aligned/'+camera][0].tobytes())),np.float32)/1000
                writer.submit_frames(camera, [DepthFrameRecord(i, raw, aligned, float(gain), 0.,
                                                               int((~invalid).sum()), raw_payload=payload)])
                if progress is not None and (i % 100 == 0 or i == len(gains)-1):
                    progress(i+1, len(gains))
            writer.mark_complete()
        finally:
            writer.close()
        with h5py.File(output, 'r+') as new:
            # Preserve exact original bytes in addition to decoded frame zero.
            new['aligned/'+camera][0] = old['aligned/'+camera][0]
            new['alignment/'+camera+'/valid_count'][0] = old['alignment/'+camera+'/valid_count'][0]
            new['alignment/'+camera+'/shift'][0] = old['alignment/'+camera+'/shift'][0]
            if not np.array_equal(new['aligned/'+camera][0], old['aligned/'+camera][0]):
                raise ValueError('Frame-zero payload changed')
    report = dict(schema='real2sim2real.materialized-depth/1', frames=len(gains),
                  camera=camera, output_sha256=digest(output), **identity,
                  human_changed=False, old_poses_approved=False, requires_fresh_tracking=True,
                  full_reference_approved=False, gains=gains.tolist())
    write_json(output.with_suffix('.json'), report)
    return report
