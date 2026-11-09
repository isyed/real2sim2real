"""Compare recorded isolated-test settings with a target controller explicitly."""
import math


def isolated_run_parameters(controller_sim):
    """Require matching solver settings for the bounded isolated-test runner.

    The runner currently implements 200Hz physics and records at50Hz. Refuse
    unsupported rates/solvers rather than silently testing another model.
    World-static support is deliberate: an isolated scene has no controller
    reset loop to pin the runtime's kinematic support body.
    """
    required=dict(solver='newton',integrator='implicitfast',cone='pyramidal',
                  iterations=100,ls_iterations=50,fps=50.,substeps=4)
    for key,expected in required.items():
        value=controller_sim.get(key)
        if isinstance(value,bool) or value!=expected:
            raise ValueError('Isolated runner does not match controller setting: '+key)
    ratio=controller_sim.get('impratio')
    if isinstance(ratio,bool) or not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or ratio<=0:
        raise ValueError('Explicit positive finite controller impratio required')
    multiccd=controller_sim.get('enable_multiccd',False)
    if type(multiccd) is not bool:raise ValueError('Explicit boolean multi-contact setting required')
    capacities={key:controller_sim[key] for key in ('nconmax','njmax') if key in controller_sim}
    if any(type(value) is not int or value<=0 for value in capacities.values()):
        raise ValueError('Positive integer solver buffer capacities required')
    return dict(impratio=float(ratio),enable_multiccd=multiccd,support_mode='world_static',**capacities)


def compare_solver_settings(summary, controller_sim):
    """Check recorded solver settings, not geometry or backend equivalence.

    Missing settings fail the comparison. An isolated stress test may still be
    useful when settings differ, but is not matching-policy settling evidence.
    Contact mode is compared when the controller records it explicitly.
    """
    recorded=summary.get('physics_parameters',{})
    keys=('solver','integrator','cone','impratio','iterations','ls_iterations')
    expected={k:controller_sim.get(k) for k in keys}
    for key in ('nconmax','njmax'):
        if key in controller_sim:
            if type(controller_sim[key]) is not int or controller_sim[key]<=0:
                raise ValueError('Positive integer solver buffer capacities required')
            expected[key]=controller_sim[key]
    fps,substeps=controller_sim.get('fps'),controller_sim.get('substeps')
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
            for v in (fps,substeps))):
        raise ValueError('Positive finite controller rate and substeps required')
    expected['timestep_s']=1/(fps*substeps)
    if 'enable_multiccd' in controller_sim:
        if type(controller_sim['enable_multiccd']) is not bool:
            raise ValueError('Explicit boolean multi-contact setting required')
        expected['enable_multiccd']=controller_sim['enable_multiccd']
    missing=[];different=[]
    for key,value in expected.items():
        if value is None or key not in recorded:
            missing.append(key);continue
        actual=recorded[key]
        if key in ('impratio','timestep_s'):
            equal=(not isinstance(actual,bool) and isinstance(actual,(int,float))
                   and math.isfinite(actual) and math.isclose(actual,value,rel_tol=1e-8,abs_tol=1e-12))
        else:equal=type(actual) is type(value) and actual==value
        if not equal:different.append(dict(setting=key,isolated=actual,controller=value))
    return dict(recorded_solver_settings_match=not missing and not different,
        compared_settings=list(expected),missing_settings=missing,differences=different,
        backend=summary.get('backend'),full_physics_equivalence_established=False,
        source_reference_or_trajectory_changed=False)
