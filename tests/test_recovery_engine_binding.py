from types import SimpleNamespace
import sys
import pytest
from real2sim2real.reconstruction import tracking_recovery,backend


def fixture(tmp_path,monkeypatch,path=None,compatible=True):
    source=tmp_path/'engine';expected=source/'prep/run_foundationpose_mhr_export.py'
    expected.parent.mkdir(parents=True);expected.write_text('fixture')
    def run(export,depth,output,*,camera_id=0,start=0,iteration=5,max_attempts=1,
            first_usable_frame_gt_rotation_oracle=False,sequence_mode=None,observed_initial_pose=None,viz_file=None):pass
    value=SimpleNamespace(__file__=str(path or expected),run_foundationpose_mhr_export=run if compatible else lambda a,b,c:None)
    monkeypatch.setattr(backend,'engine_paths',lambda:(source,tmp_path/'body',tmp_path/'fp'))
    monkeypatch.setattr('importlib.import_module',lambda name:value)
    monkeypatch.setattr(sys,'path',sys.path.copy())
    return value,source


def test_explicit_source_precedes_container_cwd_and_api_is_bound(tmp_path,monkeypatch):
    value,source=fixture(tmp_path,monkeypatch)
    assert tracking_recovery.load_exporter() is value
    assert sys.path[0]==str(source)


def test_cached_wrong_exporter_is_rejected(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,path=tmp_path/'old_container_exporter.py')
    with pytest.raises(ValueError,match='outside the selected'):tracking_recovery.load_exporter()


def test_missing_observed_seed_api_is_rejected_before_tracking(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,compatible=False)
    with pytest.raises(TypeError):tracking_recovery.load_exporter()
