"""Audit task effectors against rewards without inventing contact labels.

Profiles are explicit, copy-on-write experiments. They do not modify an active
learner, grant training admission or establish physical task success.
"""
from copy import deepcopy
import numpy as np


def _roles(expected_effectors):
    if (isinstance(expected_effectors,str) or not expected_effectors
            or any(not isinstance(x,str) or not x.strip() for x in expected_effectors)
            or len(set(expected_effectors))!=len(expected_effectors)):
        raise ValueError('Declare unique nonempty task effectors')
    return tuple(expected_effectors)


def _uses_hands(roles):
    return any('hand' in r.lower() or 'gripper' in r.lower() for r in roles)


def audit(config, hand_contact_active, *, expected_effectors, control_fps):
    roles=_roles(expected_effectors)
    phases=np.asarray(hand_contact_active)
    if phases.ndim!=2 or not phases.size or not np.isfinite(phases).all() or not np.isin(phases,[0,1]).all():
        raise ValueError('Expected explicit binary [hand, frame] phases')
    if isinstance(control_fps,bool) or not np.isfinite(control_fps) or control_fps<=0:
        raise ValueError('Expected positive control FPS')
    objective=config['objective'];collision=config['collision']
    def weight(name):
        value=float(objective.get(name+'_weight',0))
        if not np.isfinite(value):raise ValueError('Nonfinite reward weight: '+name)
        return value
    hand_task=_uses_hands(roles)
    count=np.count_nonzero(phases,axis=1)
    enabled=int(np.any(phases,axis=0).sum())
    issues=[]
    if not collision.get('robot_object_collision',False):issues.append('robot_object_collision_disabled')
    if not hand_task and collision.get('robot_scope')!='all':issues.append('nonhand_collision_scope_missing')
    if hand_task and not enabled:issues.append('required_hand_phases_missing')
    hand_bonus=weight('force_closure')
    if not hand_task and enabled and hand_bonus>0:issues.append('incidental_hand_grasp_reward_on_nonhand_task')
    if not hand_task:issues.append('hand_contact_diagnostics_do_not_measure_task_effector_contact')
    return dict(schema='real2sim2real.reward-alignment/1',expected_effectors=list(roles),
        hand_task=hand_task,hand_phase_frames_by_channel=count.tolist(),
        any_hand_phase_frames=enabled,control_frames=int(phases.shape[1]),
        hand_bonus_enabled_seconds=enabled/float(control_fps),
        object_outcome_reward_present=bool(weight('object_position')>0 or weight('object_orientation')>0),
        force_closure_weight=hand_bonus,
        hand_grasp_reward_active=bool(enabled and hand_bonus>0),
        specific_nonhand_contact_reward_implemented=False,
        issues=issues,config_changed=False,training_admission_established=False,
        physical_success_established=False,
        scope='ReconBody hand-force-closure/whole-body/object tracking; no dedicated nonhand contact term')


def without_hand_grasp_reward(config, *, expected_effectors):
    """Explicit fresh-run profile for a nonhand task; preserve posture tracking.

    Disables only hand force closure in the base objective and every curriculum
    stage. Object, pelvis, palm and body rewards remain unchanged. This does not
    add a foot reward or edit geometric reference contact phases.
    """
    roles=_roles(expected_effectors)
    if _uses_hands(roles):raise ValueError('Cannot remove grasp reward from a declared hand task')
    training=config['training'];checkpoint=training.get('checkpoint',{})
    if training.get('resume') or training.get('checkpoint_path') or checkpoint.get('load_replay_path'):
        raise ValueError('A changed reward profile requires a fresh learner and replay')
    if config['objective']!=config['env']['objective'] or config['curriculum']!=training['curriculum']:
        raise ValueError('Runtime objective/curriculum copies differ')
    if not config['collision'].get('robot_object_collision') or config['collision'].get('robot_scope')!='all':
        raise ValueError('Enable nonhand robot/object collisions before using this profile')
    result=deepcopy(config)
    result['objective']['force_closure_weight']=0.
    result['env']['objective']=deepcopy(result['objective'])
    for stage in result['curriculum']['stages']:
        stage.setdefault('objective_weights',{})['force_closure']=0.
    result['training']['curriculum']=deepcopy(result['curriculum'])
    return result
