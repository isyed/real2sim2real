import json,time
import pytest
from real2sim2real.gpu_sharing import register,owner_finished,borrower_finished,close_if_idle,active_borrowers


def lease(tmp_path):
    (tmp_path/'clock_session.json').write_text(json.dumps(dict(status='active',started_epoch=time.time(),max_seconds=3600)))
    return tmp_path


def test_owner_cannot_restore_while_borrower_needs_clock_cap(tmp_path):
    s=lease(tmp_path);register(s,'mask_job',seconds=600)
    assert not owner_finished(s)
    assert not (s/'clock_session.done').exists()
    assert borrower_finished(s,'mask_job')
    assert (s/'clock_session.done').exists()


def test_borrower_finishing_first_preserves_owner_session(tmp_path):
    s=lease(tmp_path);register(s,'mask_job',seconds=600)
    assert not borrower_finished(s,'mask_job')
    assert not (s/'clock_session.done').exists()
    assert owner_finished(s)


def test_expired_borrower_does_not_hold_clock_until_owner_deadline(tmp_path):
    s=lease(tmp_path);r=register(s,'mask_job',seconds=600);owner_finished(s)
    assert not active_borrowers(s,now=r['deadline_epoch']+1)
    path=s/'borrowers/mask_job.json';record=json.loads(path.read_text());record['deadline_epoch']=time.time()-1;path.write_text(json.dumps(record))
    assert close_if_idle(s)


def test_completed_or_insufficient_session_rejects_new_gpu_work(tmp_path):
    s=lease(tmp_path);owner_finished(s)
    with pytest.raises(ValueError):register(s,'late_job',seconds=600)


def test_old_owner_completion_cannot_close_a_new_lease(tmp_path):
    s=lease(tmp_path)
    (s/'owner_gpu_finished.json').write_text(json.dumps(dict(lease_started_epoch=0)))
    assert not close_if_idle(s)
