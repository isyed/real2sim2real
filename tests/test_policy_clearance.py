import numpy as np
import pytest
from real2sim2real.controller_actor_training.policy_clearance import summarize


def contact(**kwargs):
    return dict(dict(shape0=0,shape1=1,point0=[0,0,0],point1=[0,0,0],
                     normal=[0,0,-1],margin0=0.,margin1=0.),**kwargs)


def test_local_contact_points_are_transformed_and_grouped():
    # The object contact is 6 mm below the table in world space.
    q=np.array([[0,0,.994,0,0,0,1],[0,0,1,0,0,0,1]])
    result=summarize(q,['object','table'],[0,1],['objects','support'],[contact()])
    row=result['contacts'][0]
    assert row['penetration_m']==pytest.approx(.006)
    assert row['point0_world_m']==pytest.approx([0,0,.994])
    assert result['groups']['objects/support']['maximum_penetration_m']==pytest.approx(.006)
    assert result['physical_integration'] is False


def test_world_static_shape_and_contact_margins():
    q=np.array([[0,0,.003,0,0,0,1]])
    result=summarize(q,['foot'],[0,-1],['robot','ground'],[contact(margin0=.002,margin1=.002)])
    row=result['contacts'][0]
    assert row['raw_signed_gap_m']==pytest.approx(.003)
    assert row['penetration_m']==pytest.approx(.001)
    assert row['body1']=='world'


def test_rotated_body_frame_contact_is_used():
    q=np.array([[0,0,0,0,0,np.sqrt(.5),np.sqrt(.5)]])
    result=summarize(q,['body'],[0,-1],['robot','support'],[
        contact(point0=[1,0,0],point1=[0,.9,0],normal=[0,1,0])])
    assert result['contacts'][0]['point0_world_m']==pytest.approx([0,1,0])
    assert result['contacts'][0]['penetration_m']==pytest.approx(.1)


@pytest.mark.parametrize('bad',[{'shape0':-1},{'normal':[0,0,-2]},
                               {'margin0':-.1},{'point0':[float('nan'),0,0]}])
def test_invalid_contact_is_rejected(bad):
    with pytest.raises(ValueError):
        summarize(np.array([[0,0,0,0,0,0,1]]),['body'],[0,-1],['objects','support'],[contact(**bad)])


def test_no_generated_contacts_is_not_global_clearance_proof():
    result=summarize(np.array([[0,0,1,0,0,0,1]]),['body'],[0],['robot'],[])
    assert result['groups']=={}
    assert 'not a global clearance proof' in result['limitation']


def test_replay_accepts_a_policy_with_omitted_optional_solver_settings():
    from real2sim2real.controller_actor_training.policy_clearance import replay_solver_options
    from flash_chord.runtime.sim import SimConfig
    options=replay_solver_options({'_target_':'flash_chord.runtime.sim.SimConfig','impratio':1.,'nconmax':512})
    assert options['enable_multiccd'] is SimConfig().enable_multiccd
    assert options['use_mujoco_contacts'] is SimConfig().use_mujoco_contacts
    assert options['impratio']==1. and options['nconmax']==512


def test_replay_preserves_explicit_contact_mode():
    from real2sim2real.controller_actor_training.policy_clearance import replay_solver_options
    options=replay_solver_options({'_target_':'flash_chord.runtime.sim.SimConfig','enable_multiccd':True,'use_mujoco_contacts':False})
    assert options['enable_multiccd'] is True and options['use_mujoco_contacts'] is False
