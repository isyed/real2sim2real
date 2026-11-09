import pytest
from real2sim2real.controller_actor_training.settling_compatibility import compare_solver_settings


def fixture():
    sim=dict(solver='newton',integrator='implicitfast',cone='pyramidal',impratio=1.,
             iterations=100,ls_iterations=50,fps=50.,substeps=4,enable_multiccd=False)
    summary=dict(backend='MuJoCo-C CPU',physics_parameters={k:v for k,v in sim.items() if k not in ('fps','substeps')})
    summary['physics_parameters']['timestep_s']=.005
    return summary,sim


def test_matching_solver_does_not_certify_backend_or_geometry():
    r=compare_solver_settings(*fixture())
    assert r['recorded_solver_settings_match'] and not r['full_physics_equivalence_established']


def test_stress_ratio_cannot_be_claimed_matching_controller():
    summary,sim=fixture();summary['physics_parameters']['impratio']=20.
    r=compare_solver_settings(summary,sim)
    assert not r['recorded_solver_settings_match']
    assert r['differences']==[dict(setting='impratio',isolated=20.,controller=1.)]


def test_missing_contact_mode_is_not_silently_assumed():
    summary,sim=fixture();del summary['physics_parameters']['enable_multiccd']
    r=compare_solver_settings(summary,sim)
    assert not r['recorded_solver_settings_match'] and r['missing_settings']==['enable_multiccd']


def test_substep_difference_changes_physics_clock():
    summary,sim=fixture();sim['substeps']=2
    assert 'timestep_s' in [v['setting'] for v in compare_solver_settings(summary,sim)['differences']]

def test_larger_isolated_contact_buffer_is_not_matching_policy_evidence():
    summary,sim=fixture();sim.update(nconmax=512,njmax=1024)
    summary['physics_parameters'].update(nconmax=4096,njmax=16384)
    result=compare_solver_settings(summary,sim)
    assert not result['recorded_solver_settings_match']
    assert {x['setting'] for x in result['differences']}=={'nconmax','njmax'}

def test_missing_policy_capacity_is_reported():
    summary,sim=fixture();sim['nconmax']=512
    assert 'nconmax' in compare_solver_settings(summary,sim)['missing_settings']


@pytest.mark.parametrize('rate',[0,True,float('nan')])
def test_invalid_control_clock_rejected(rate):
    summary,sim=fixture();sim['fps']=rate
    with pytest.raises(ValueError):compare_solver_settings(summary,sim)
