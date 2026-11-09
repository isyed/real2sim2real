from types import SimpleNamespace
import pytest
from real2sim2real.controller_actor_training.isolated_object_stability import apply_scene_parameters


def builder(dofs=6):
    return SimpleNamespace(joint_dof_count=dofs,shape_count=3,shape_flags=[2,0,2],shape_material_mu=[.5,.7,.5],
        custom_attributes={'mujoco:dof_passive_damping':SimpleNamespace(values={i:1. for i in range(6)})})


def test_explicit_parity_covers_colliding_surfaces_and_all_free_object_dofs():
    b=builder();p=dict(contact_friction=1.,object_free_joint_damping=0.)
    assert apply_scene_parameters(b,p,collide_flag=2)==p
    assert b.shape_material_mu==[1.,.7,1.]
    assert list(b.custom_attributes['mujoco:dof_passive_damping'].values.values())==[0.]*6


def test_another_scene_cannot_silently_change_its_joints():
    b=builder(12)
    with pytest.raises(ValueError):apply_scene_parameters(b,dict(contact_friction=1.,object_free_joint_damping=0.),collide_flag=2)
    assert b.shape_material_mu==[.5,.7,.5]


@pytest.mark.parametrize('parameters',[{},dict(contact_friction=float('nan'),object_free_joint_damping=0.),dict(contact_friction=1.,object_free_joint_damping=-1.)])
def test_invalid_or_incomplete_material_request_is_rejected(parameters):
    b=builder()
    with pytest.raises(ValueError):apply_scene_parameters(b,parameters,collide_flag=2)
    assert b.shape_material_mu==[.5,.7,.5]
