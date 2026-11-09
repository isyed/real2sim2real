from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training.unassisted_pilot import configure


def config():
    curriculum=dict(stages=[dict(voc_scale=1.,immediate_first_frame_probability=.25),dict(voc_scale=.5,immediate_first_frame_probability=.5)])
    return dict(curriculum=curriculum,training=dict(curriculum=deepcopy(curriculum),replay=dict(minimum_size=100000,batch_size=8192,capacity=1000000)),voc_scale_override=None)


def test_all_assistance_routes_zero_without_mutating_input():
    original=config();c,r=configure(original)
    assert c['voc_scale_override']==0 and r['curriculum_assistance_scales']==[0.,0.]
    assert c['curriculum']==c['training']['curriculum']
    assert all(s['immediate_first_frame_probability']==1. for s in c['curriculum']['stages'])
    assert original['voc_scale_override'] is None and original['training']['replay']['capacity']==1000000


def test_replay_must_hold_warmup_and_batch():
    with pytest.raises(ValueError):configure(config(),replay_capacity=1024)
