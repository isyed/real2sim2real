import pytest
from real2sim2real.reconstruction.tracking_runtime import MODEL_RUNS,tracking_environment


def test_explicit_tracking_model_files_required(tmp_path):
    with pytest.raises(FileNotFoundError):tracking_environment(tmp_path)
    for run in MODEL_RUNS:
        d=tmp_path/run;d.mkdir()
        (d/'config.yml').write_text('test: true\n');(d/'model_best.pth').write_bytes(b'fixture')
    env=tracking_environment(tmp_path)
    assert env['FOUNDATIONPOSE_WEIGHTS_DIR']==str(tmp_path.resolve())
    assert env['R2S2R_FP_RENDER_BATCH']=='16'
    (tmp_path/MODEL_RUNS[1]/'model_best.pth').write_bytes(b'')
    with pytest.raises(FileNotFoundError):tracking_environment(tmp_path)
