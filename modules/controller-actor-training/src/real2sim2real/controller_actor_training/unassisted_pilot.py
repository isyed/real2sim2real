"""Explicit zero-object-assistance mode for bounded controller diagnostics."""
from copy import deepcopy


def configure(configuration,*,replay_capacity=131072,frame_zero_probability=1.):
    if type(replay_capacity) is not int or replay_capacity<1 or not 0<=frame_zero_probability<=1:
        raise ValueError('Finite bounded replay and reset probability required')
    c=deepcopy(configuration)
    if c['curriculum']!=c['training']['curriculum']:raise ValueError('Curriculum views differ')
    replay=c['training']['replay']
    if replay_capacity<max(replay['minimum_size'],replay['batch_size']):raise ValueError('Replay cannot hold warmup/batch')
    replay['capacity']=replay_capacity
    c['voc_scale_override']=0.
    for curriculum in [c['curriculum'],c['training']['curriculum']]:
        for stage in curriculum['stages']:
            stage['voc_scale']=0.;stage['immediate_first_frame_probability']=float(frame_zero_probability)
    return c,dict(schema='real2sim2real.unassisted-pilot/1',object_assistance_override=0.,
        curriculum_assistance_scales=[s['voc_scale'] for s in c['curriculum']['stages']],
        frame_zero_probability=frame_zero_probability,replay_capacity=replay_capacity,
        body_controller_retained=True,hand_controller_retained=True,task_success_established=False)
