import pytest
from flash_chord.runtime.sim import SimConfig,build_solver


def test_legacy_default_and_explicit_multiccd_reach_solver(monkeypatch):
    newton = pytest.importorskip("newton", reason="Requires the optional simulation runtime")
    captured=[]
    monkeypatch.setattr(newton.solvers,'SolverMuJoCo',lambda model,**kw:captured.append(kw))
    model=object()
    build_solver(model,SimConfig())
    build_solver(model,SimConfig(enable_multiccd=True))
    assert captured[0]['enable_multiccd'] is False
    assert captured[1]['enable_multiccd'] is True
    assert {k:v for k,v in captured[0].items() if k!='enable_multiccd'}=={k:v for k,v in captured[1].items() if k!='enable_multiccd'}


@pytest.mark.parametrize('value',[1,'true',None])
def test_multiccd_is_never_enabled_by_truthy_nonboolean_value(value):
    with pytest.raises(ValueError,match='boolean'):SimConfig(enable_multiccd=value)


@pytest.mark.parametrize('value',[True,0.,-1.,float('nan'),float('inf')])
def test_isolated_solver_ratio_rejects_invalid_values_before_execution(tmp_path,monkeypatch,value):
    from real2sim2real.controller_actor_training.isolated_object_stability import run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError):
        run(tmp_path/'unused.urdf',[0,0,1,0,0,0,1],tmp_path/'not_created',
            enable_multiccd=True,impratio=value)
    assert not (tmp_path/'not_created').exists()
