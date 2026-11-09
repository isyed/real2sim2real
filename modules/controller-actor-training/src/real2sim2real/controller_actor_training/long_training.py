"""Fresh or exactly resumed long residual training on a portable policy.

Object assistance fades during the first ten million transitions only. Reset
settling cannot reintroduce assistance after that boundary. Evaluation starts
at native frame zero without assistance at every checkpoint.
"""
from copy import deepcopy
from .continuation_curriculum import continuation_review

CEILING=60_000_000
EVALUATIONS=(5_000_000,10_000_000,15_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000)


def configuration(policy,output,*,environment_steps,worlds=256,
                  resume_state=None,resume_replay=None,orientation_weight=.1):
    from .residual import configuration as residual_configuration
    from .frozen import materialize
    if type(environment_steps) is not int or not 5_000_000<=environment_steps<=CEILING or environment_steps%5_000_000:
        raise ValueError('Explicit five-million checkpoint target through60M required')
    _,frozen=materialize(policy)
    c=residual_configuration(policy,output,environment_steps=environment_steps,worlds=worlds,
        orientation_weight=orientation_weight,resume_state=resume_state,resume_replay=resume_replay,
        assistance_thresholds=[4_000_000,7_000_000,10_000_000,CEILING])
    # The portable policy owns the tested geometry, materials, contact settings
    # and solver. A fresh RL preset must not silently replace these settings.
    c['scene']=deepcopy(frozen['scene']);c['scene']['world_count']=worlds
    c['sim']=deepcopy(frozen['env']['sim']);c['env']['sim']=deepcopy(c['sim'])
    c['collision']=deepcopy(frozen['collision'])
    template=c['curriculum']['stages'][-1]
    stages=[]
    for scale,probability in [(1.,.5),(.5,.5),(.1,.5),(0.,.5),(0.,.75)]:
        stage=deepcopy(template)
        stage.update(voc_scale=scale,immediate_first_frame_probability=probability)
        stages.append(stage)
    # Round down to a full vector collection: assistance never extends past10M.
    c['curriculum'].update(thresholds=[v//worlds*worlds for v in [4_000_000,7_000_000,10_000_000,40_000_000,CEILING]],stages=stages)
    c['training']['curriculum']=deepcopy(c['curriculum'])
    # Recovery resets may freeze/settle, but must not inject their old VOC=1
    # helper during the supposedly unassisted phase. During assisted stages
    # their existing fade interpolates from zero to the current task scale.
    c['reset']['reset_voc_scale']=0.
    c['env']['reset']=deepcopy(c['reset'])
    return c


def review(baseline,previous,current,*,target):
    if target not in EVALUATIONS:raise ValueError('Unknown declared evaluation milestone')
    if not current.get('valid'):return dict(continue_training=False,reason='invalid_evaluation')
    if target<15_000_000:
        return dict(continue_training=True,reason='diagnostic_before_first_15m_gate',diagnostic_only=True)
    return continuation_review(baseline,previous,current,major_review=target==40_000_000)
