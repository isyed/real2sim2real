import shutil
import pytest
from real2sim2real.artifacts import publish
from real2sim2real.controller_actor_training import frozen


def test_recipe_selects_flash_sac_and_has_no_assistance_or_actor_resume():
    cfg=frozen.configuration()
    assert cfg['training']['_target_']=='flash_chord.training.flash_sac.config.TrainingConfig'
    assert cfg['training']['curriculum'] is None
    assert cfg['env']['voc']['max_force']==cfg['env']['voc']['max_torque']==0
    assert cfg['action']['hand_closing']


@pytest.mark.parametrize('change',['force','scale','hands','resume','ppo'])
def test_controller_rejects_isolation_regressions(change):
    c=frozen.configuration()
    if change=='force':c['env']['voc']['max_force']=60
    if change=='scale':c['voc_scale_override']=1
    if change=='hands':c['action']['hand_closing']=None
    if change=='resume':c['training']['checkpoint_path']='an-actor.safetensors'
    if change=='ppo':c['training']['_target_']='flash_chord.training.ppo.config.TrainingConfig'
    with pytest.raises(ValueError):frozen.validate_config(c)


def test_bundle_remains_usable_after_source_removal_and_relocation(tmp_path):
    reference=tmp_path/'input';reference.mkdir();(reference/'reference.parquet').write_bytes(b'test-reference')
    publish(reference,'robot-reference',{'reference':reference/'reference.parquet'},metadata=dict(robot='g1'))
    assets=tmp_path/'assets'
    for name in ['policies/sonic/encoder_batched.onnx','policies/sonic/decoder_batched.onnx','urdfs/g1/main_with_hand.urdf']:
        path=assets/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
    (assets/'urdfs/g1/main_with_hand.urdf').write_text('<robot name="test"><link name="body"><visual><geometry><mesh filename="../../meshes/g1/body.obj"/></geometry></visual></link></robot>')
    mesh=assets/'meshes/g1/body.obj';mesh.parent.mkdir(parents=True);mesh.write_text('v 0 0 0\n')
    output=frozen.package(reference,assets,tmp_path/'policy',hardlink=True)
    shutil.rmtree(reference);shutil.rmtree(assets)
    moved=tmp_path/'relocated';output.rename(moved)
    manifest,cfg=frozen.materialize(moved)
    assert cfg['task']['parquet']==str(moved/'reference/reference.parquet')
    assert cfg['action']['policy_dir']==str(moved/'assets/policies/sonic')
    assert not manifest['actor_present']
    assert (moved/'assets/meshes/g1/body.obj').read_text()=='v 0 0 0\n'
    (moved/'assets/policies/sonic/encoder_batched.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):frozen.materialize(moved)


def test_optional_collision_cache_is_copied_validated_and_independent(tmp_path):
    import numpy as np
    source=tmp_path/'cache';source.mkdir();target=tmp_path/'copied'
    key='0123456789abcdef.npz';path=source/key
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    np.savez(path,n=np.int64(1),v0=vertices,f0=np.array([[0,1,2]],np.int32))
    assert frozen.copy_collision_cache(source,target)==[key]
    shutil.rmtree(source)
    with np.load(target/key) as data:np.testing.assert_array_equal(data['v0'],vertices)
    bad=tmp_path/'bad';bad.mkdir()
    np.savez(bad/key,n=np.int64(1),v0=vertices,f0=np.array([[0,1,9]],np.int32))
    with pytest.raises(ValueError,match='geometry'):frozen.copy_collision_cache(bad,tmp_path/'rejected')
