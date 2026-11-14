import json
import numpy as np
import pytest

from real2sim2real.mujoco_assets import load_model, path_mapping, relocated_path, relocated_scene


def test_mapping_requires_component_boundary():
    assert relocated_path('/old/assets/a', {'/old/assets': '/new'}) == '/new/a'
    assert relocated_path('/old/assets2/a', {'/old/assets': '/new'}) == '/old/assets2/a'


def test_reject_nonabsolute_mapping(tmp_path, monkeypatch):
    p=tmp_path/'map.json';p.write_text(json.dumps({'relative':'/new'}))
    monkeypatch.setenv('R2S2R_ASSET_PATH_MAP',str(p))
    with pytest.raises(ValueError,match='absolute'):path_mapping()


def test_nested_include_preserves_original_and_cleans_up(tmp_path):
    child=tmp_path/'child.xml';child.write_text('<mujocoinclude><asset><mesh file="/old/mesh.stl"/></asset></mujocoinclude>')
    source=tmp_path/'scene.xml';source.write_text('<mujoco><include file="child.xml"/><worldbody/></mujoco>')
    before={p:p.read_bytes() for p in [source,child]}
    with relocated_scene(source,{'/old':'/new'}) as p:
        assert p!=source
        assert '/new/mesh.stl' in next(q for q in tmp_path.glob('.r2s2r-assets-*') if '/new/' in q.read_text()).read_text()
    assert not list(tmp_path.glob('.r2s2r-assets-*'))
    assert all(p.read_bytes()==data for p,data in before.items())


def test_compile_matches_original_after_asset_move(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    # Inline mesh has no filesystem asset; compiler path still exercises relocation.
    source=tmp_path/'scene.xml'
    source.write_text(f'<mujoco><compiler meshdir="{old}"/><worldbody><body><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/></body></worldbody></mujoco>')
    expected=mujoco.MjModel.from_xml_path(str(source))
    before=source.read_bytes();old.rmdir()
    actual=load_model(source,mapping={str(old):str(new)})
    for name in ['body_mass','body_inertia','geom_size','qpos0','jnt_range']:
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    assert source.read_bytes()==before and not list(tmp_path.glob('.r2s2r-assets-*'))
