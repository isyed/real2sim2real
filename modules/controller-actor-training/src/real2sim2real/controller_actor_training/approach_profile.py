"""Explicit coarse approach shaping for fresh tracking-policy experiments.

Broader position kernels retain a useful reward away from a grasp target. This
does not repair source geometry, guarantee locomotion or establish grasp success.
"""
from copy import deepcopy
import math


def configure(configuration, *, pelvis_std_m=.5,palm_std_m=.4,
              pelvis_weight=2.,palm_weight=2.,pelvis_orientation_weight=2.,prior_configuration=None):
    values=[pelvis_std_m,palm_std_m,pelvis_weight,palm_weight,pelvis_orientation_weight]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Explicit finite positive shaping scales and weights required')
    resume=bool(configuration['training'].get('resume'))
    if resume and prior_configuration is None:
        raise ValueError('Changed approach shaping requires a fresh learner or verified matching profile')
    result=deepcopy(configuration)
    if result['objective']!=result['env']['objective'] or result['curriculum']!=result['training']['curriculum']:
        raise ValueError('Objective/curriculum views must agree before applying a profile')
    updates=dict(pelvis_position_std=pelvis_std_m,palm_position_std=palm_std_m,
        pelvis_position_weight=pelvis_weight,palm_position_weight=palm_weight,
        pelvis_orientation_weight=pelvis_orientation_weight)
    if any(k not in result['objective'] for k in updates):raise ValueError('Unsupported tracking objective')
    before={k:result['objective'][k] for k in updates}
    for objective in [result['objective'],result['env']['objective']]:objective.update(updates)
    weights=dict(pelvis_position=pelvis_weight,palm_position=palm_weight,pelvis_orientation=pelvis_orientation_weight)
    for curriculum in [result['curriculum'],result['training']['curriculum']]:
        for stage in curriculum['stages']:
            if any(k not in stage['objective_weights'] for k in weights):raise ValueError('Missing stage tracking weight')
            stage['objective_weights'].update(weights)
    if resume:
        # A new objective cannot silently reuse optimizer/replay from the old
        # profile. Check the actual checkpoint config supplied by the caller.
        for key in ['objective','curriculum','task','action','observation','sim','scene']:
            if key not in prior_configuration or prior_configuration[key]!=result.get(key):
                raise ValueError('Resume configuration differs from requested approach profile: '+key)
    return result,dict(schema='real2sim2real.approach-reward-profile/1',previous=before,requested=updates,
        fresh_learner_required=True,geometry_changed=False,reference_changed=False,action_or_observation_changed=False,
        contact_phases_changed=False,orientation_task_guidance_changed=False,physical_improvement_established=False,
        matching_profile_resume_verified=resume)


def position_reward(squared_error,std_m):
    """Numerical audit of the runtime exponential position reward."""
    import numpy as np
    e=np.asarray(squared_error,float)
    if not np.isfinite(e).all() or np.any(e<0) or isinstance(std_m,bool) or not math.isfinite(std_m) or std_m<=0:
        raise ValueError('Finite nonnegative squared errors and positive metric scale required')
    return np.exp(-e/(std_m*std_m))
