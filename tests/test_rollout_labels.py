from real2sim2real.controller_actor_training.render_frozen import rollout_label


def test_comparison_labels_distinguish_recorded_actor_and_explicit_override():
    assert 'Frozen SONIC' in rollout_label({'actor_loaded':False})
    assert 'Learned actor' in rollout_label({'actor_loaded':True})
    assert 'bounded joint correction' in rollout_label({'actor_loaded':True,'controller_override':{'values':{}}})
    assert 'bounded joint correction' not in rollout_label({'actor_loaded':True})
