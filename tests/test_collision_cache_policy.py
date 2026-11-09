import json
import numpy as np
import pytest
from real2sim2real.artifacts import publish,load
from real2sim2real.controller_actor_training import collision_cache_policy as module

def fixture(tmp_path,monkeypatch):
    source=tmp_path/'source';(source/'reference').mkdir(parents=True)
    (source/'reference/native.parquet').write_bytes(b'native-timeline')
    (source/'reference/control.parquet').write_bytes(b'already-prepared-clock')
    manifest=dict(reference='reference/control.parquet',native_reference='reference/native.parquet',collision_cache_files=[],config={})
    (source/'manifest.json').write_text(json.dumps(manifest))
    publish(source,'frozen-controller',dict(manifest=source/'manifest.json',native=source/'reference/native.parquet',control=source/'reference/control.parquet'),metadata=dict(physics_validated=False))
    monkeypatch.setattr(module,'materialize',lambda p:(json.loads((p/'manifest.json').read_text()),{}))
    cache=tmp_path/'cache';cache.mkdir()
    np.savez(cache/'0123456789abcdef.npz',n=np.array(1),v0=np.array([[0.,0,0],[1,0,0],[0,1,0]]),f0=np.array([[0,1,2]]))
    return source,cache

def test_pinning_does_not_rebuild_existing_control_clock_or_mutate_parent(tmp_path,monkeypatch):
    source,cache=fixture(tmp_path,monkeypatch);original=(source/'manifest.json').read_bytes()
    out=tmp_path/'pinned';r=module.package(source,cache,out,evidence={'source':'verified prior hull'},hardlink=True)
    assert (source/'manifest.json').read_bytes()==original
    assert (out/'reference/control.parquet').read_bytes()==b'already-prepared-clock'
    assert json.loads((out/'manifest.json').read_text())['collision_cache_files']==['0123456789abcdef.npz']
    assert not r['reference_clock_rebuilt'] and not r['physical_equivalence_established']
    load(source,kinds={'frozen-controller'});load(out,kinds={'frozen-controller'})

def test_conflicting_existing_collision_key_is_rejected(tmp_path,monkeypatch):
    source,cache=fixture(tmp_path,monkeypatch);dest=source/'assets/.cache/convex';dest.mkdir(parents=True)
    (dest/'0123456789abcdef.npz').write_bytes(b'different-hulls')
    with pytest.raises(ValueError,match='different hull'):module.package(source,cache,tmp_path/'out',evidence={'source':'reviewed'},hardlink=True)
    load(source,kinds={'frozen-controller'})
