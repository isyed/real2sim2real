"""Explicitly retain frozen-policy scene physics in a fresh actor recipe."""
from copy import deepcopy
import math


def match_contact_mode(configuration,policy_config):
    """Explicitly adopt a physically tested policy's native contact mode.

    Only the multiple-contact boolean changes. Use this on a fresh learner or
    after a separate continuation review; it does not approve a resume. Legacy
    defaults, scene parameters, rewards and world counts are preserved.
    """
    expected=policy_config['sim'] if 'sim' in policy_config else policy_config['env']['sim']
    requested=expected.get('enable_multiccd',False)
    if type(requested) is not bool:raise ValueError('Policy contact mode must be boolean')
    if ('sim' in policy_config and 'env' in policy_config
        and policy_config['env']['sim'].get('enable_multiccd',False)!=requested):
        raise ValueError('Policy root/environment contact modes disagree')
    sims=[configuration['env']['sim']]
    if 'sim' in configuration:sims.append(configuration['sim'])
    before=[sim.get('enable_multiccd',False) for sim in sims]
    if any(type(v) is not bool for v in before) or len(set(before))!=1:
        raise ValueError('Configuration root/environment contact modes must agree')
    result=deepcopy(configuration);result['env']['sim']['enable_multiccd']=requested
    if 'sim' in result:result['sim']['enable_multiccd']=requested
    return result,dict(previous_enable_multiccd=before[0],enable_multiccd=requested,
        changed=before[0]!=requested,opt_in=True,scene_parameters_changed=False,
        reward_or_curriculum_changed=False,resume_compatibility_established=False)


def match_scene(configuration,policy_config):
    """Copy material/damping only; preserve world count,rewards and solver.

    This is opt-in and never changes the residual recipe's legacy defaults.
    A caller resuming learning must already bind the learner to this policy.
    Solver differences need a separate experiment, not a silent correction.
    """
    fields=('contact_friction','object_free_joint_damping')
    scene=policy_config['scene'];parameters={key:scene[key] for key in fields}
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in parameters.values()):
        raise ValueError('Explicit finite nonnegative policy scene parameters required')
    expected=policy_config['sim'] if 'sim' in policy_config else policy_config['env']['sim']
    root_sim=configuration['sim'] if 'sim' in configuration else configuration['env']['sim']
    for sim in [root_sim,configuration['env']['sim']]:
        for key in ['solver','integrator','cone','impratio','iterations','ls_iterations','fps','substeps']:
            if sim.get(key)!=expected.get(key) or key not in expected:raise ValueError('Solver mismatch:'+key)
        if sim.get('enable_multiccd',False)!=expected.get('enable_multiccd',False):raise ValueError('Contact-mode mismatch')
    result=deepcopy(configuration);before={key:result['scene'].get(key) for key in fields}
    result['scene'].update(parameters)
    return result,dict(policy_scene_parameters=parameters,previous_scene_parameters=before,
        changed_fields=[key for key in fields if before[key]!=parameters[key]],
        world_count_unchanged=result['scene']['world_count']==configuration['scene']['world_count'],
        solver_changed=False,reward_or_curriculum_changed=False,opt_in=True)
