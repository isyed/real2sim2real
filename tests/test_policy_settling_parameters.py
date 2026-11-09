import pytest
from real2sim2real.controller_actor_training.settling_compatibility import isolated_run_parameters

def policy():
    return dict(solver='newton',integrator='implicitfast',cone='pyramidal',iterations=100,
        ls_iterations=50,fps=50.,substeps=4,impratio=1.)

def test_controller_ratio_is_not_replaced_by_stress_default():
    assert isolated_run_parameters(policy())==dict(impratio=1.,enable_multiccd=False,support_mode='world_static')

@pytest.mark.parametrize('key,value',[('fps',60.),('substeps',2),('solver','cg'),('iterations',50),
    ('impratio',None),('impratio',True),('impratio',float('nan')),('enable_multiccd','false')])
def test_unknown_or_incompatible_settings_are_rejected(key,value):
    config=policy();config[key]=value
    with pytest.raises(ValueError):isolated_run_parameters(config)

def test_explicit_stress_test_can_still_be_requested():
    config=policy();config.update(impratio=20.,enable_multiccd=True)
    assert isolated_run_parameters(config)['impratio']==20
    assert isolated_run_parameters(config)['enable_multiccd'] is True

def test_policy_contact_capacities_are_preserved():
    config=dict(policy(),nconmax=512,njmax=1024)
    params=isolated_run_parameters(config)
    assert params['nconmax']==512 and params['njmax']==1024

@pytest.mark.parametrize('value',[0,-1,True,512.5])
def test_invalid_contact_capacity_is_rejected(value):
    with pytest.raises(ValueError):isolated_run_parameters(dict(policy(),nconmax=value))
