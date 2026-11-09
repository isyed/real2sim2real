"""Explicit inference-only bounded reference-pull ablations; checkpoint unchanged."""
from copy import deepcopy

def apply_reference_pull(config,override):
    from flash_chord.runtime.actions.sonic import SonicJointResidualActionConfig
    allowed={'reference_pull_gain','reference_pull_max_rad','reference_pull_joint_names'}
    if not isinstance(override,dict) or set(override)!=allowed:
        raise ValueError('Declare gain, bound and joint selection explicitly; no other overrides allowed')
    if override['reference_pull_joint_names'] is None:
        raise ValueError('Inference ablations require an explicit joint subset')
    # Validate values with the same runtime configuration contract.
    SonicJointResidualActionConfig(**override)
    result=deepcopy(config)
    if result['action']!=result['env']['action']:raise ValueError('Controller configurations differ')
    result['action'].update(override);result['env']['action']=deepcopy(result['action'])
    return result
