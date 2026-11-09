"""Cooperative completion of a bounded clock lease shared by local jobs.

This module never changes hardware settings or bypasses a thermal guard. The
privileged lease helper remains the sole clock owner and enforces its expiry.
Callers must already have authorization to share the device.
"""
import argparse,json,time
from pathlib import Path


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def register(session,name,*,seconds):
    session=Path(session)
    if not name.replace('_','').isalnum() or seconds<=0:raise ValueError('Bounded named borrower required')
    lease=json.loads((session/'clock_session.json').read_text())
    if lease.get('status')!='active' or (session/'clock_session.done').exists():raise ValueError('Lease is not active')
    now=time.time();deadline=min(now+seconds,lease['started_epoch']+lease['max_seconds']-60)
    if deadline-now<60:raise ValueError('Lease has insufficient time')
    receipt=dict(name=name,active=True,registered_epoch=now,deadline_epoch=deadline,
        lease_started_epoch=lease['started_epoch'],clock_owner=False)
    write(session/'borrowers'/f'{name}.json',receipt);return receipt


def active_borrowers(session,*,now=None):
    session=Path(session);now=time.time() if now is None else now
    lease=json.loads((session/'clock_session.json').read_text());active=[]
    for path in (session/'borrowers').glob('*.json'):
        value=json.loads(path.read_text())
        if value['lease_started_epoch']==lease['started_epoch'] and value['active'] and value['deadline_epoch']>now:active.append(value)
    return active


def close_if_idle(session):
    session=Path(session);lease=json.loads((session/'clock_session.json').read_text());owner=session/'owner_gpu_finished.json'
    if lease.get('status')!='active':return False
    if not owner.exists() or json.loads(owner.read_text())['lease_started_epoch']!=lease['started_epoch']:return False
    if active_borrowers(session):return False
    (session/'clock_session.done').write_text('Owner and all bounded borrowers finished GPU work\n');return True


def owner_finished(session):
    session=Path(session);lease=json.loads((session/'clock_session.json').read_text())
    write(session/'owner_gpu_finished.json',dict(lease_started_epoch=lease['started_epoch'],finished_epoch=time.time()))
    return close_if_idle(session)


def borrower_finished(session,name):
    session=Path(session);path=session/'borrowers'/f'{name}.json';record=json.loads(path.read_text())
    record.update(active=False,finished_epoch=time.time());write(path,record);return close_if_idle(session)


def watch(session):
    session=Path(session);lease=json.loads((session/'clock_session.json').read_text());deadline=lease['started_epoch']+lease['max_seconds']
    while time.time()<deadline:
        if (session/'clock_session.done').exists() or close_if_idle(session):return
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,required=True);args=parser.parse_args();watch(args.session)
