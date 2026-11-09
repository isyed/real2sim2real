from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training import bounded_approach as pilot


@pytest.fixture
def portable(monkeypatch, tmp_path):
    from real2sim2real.controller_actor_training import frozen
    c = frozen.configuration()
    c['task']['parquet'] = str(tmp_path / 'reference.parquet')
    c['action']['policy_dir'] = str(tmp_path / 'assets/policies/sonic')
    c['env']['action'] = deepcopy(c['action'])
    monkeypatch.setattr(frozen, 'materialize', lambda p: ({}, deepcopy(c)))
    return tmp_path, c


def test_full_reference_recipe_keeps_tested_scene_and_assistance_cutoff(portable):
    pytest.importorskip("newton", reason="Instantiates optional simulator configuration")
    path, frozen = portable
    c, proof = pilot.configuration(path, path / 'train', environment_steps=5_000_000)
    from real2sim2real.controller_actor_training.residual import validate
    check = validate(c)
    assert c['task'] == frozen['task'] and c['action'] == frozen['action']
    assert c['collision'] == frozen['collision']
    assert c['scene'] == dict(frozen['scene'], world_count=256)
    assert check['replay_capacity'] == 262144 and not check['resume']
    assert check['assistance_scales'] == [1., .5, .1, 0., 0.]
    assert proof['assistance_zero_from'] == 9999872
    assert c['reset']['reset_voc_scale'] == 0
    assert all(w['pelvis_position'] == 2 and w['palm_position'] == 2 and
               w['pelvis_orientation'] == 2 and w['object_orientation'] == .1
               for w in check['stage_weights'])


def test_exact_same_recipe_may_resume_but_replay_change_fails(portable):
    path, _ = portable
    prior, _ = pilot.configuration(path, path / 'train', environment_steps=5_000_000)
    c, _ = pilot.configuration(path, path / 'train', environment_steps=10_000_000,
                               resume_state='state', resume_replay='replay',
                               prior_configuration=prior)
    assert c['training']['resume']
    with pytest.raises(ValueError, match='recipe changed'):
        pilot.configuration(path, path / 'train', environment_steps=10_000_000,
                            replay_capacity=131072, resume_state='state',
                            resume_replay='replay', prior_configuration=prior)


@pytest.mark.parametrize('kwargs', [dict(environment_steps=20_000_000),
    dict(environment_steps=True), dict(environment_steps=5_000_000, replay_capacity=99999),
    dict(environment_steps=10_000_000, resume_state='state', resume_replay='replay')])
def test_invalid_targets_buffers_and_unverified_resume_rejected(portable, kwargs):
    path, _ = portable
    with pytest.raises(ValueError):
        pilot.configuration(path, path / 'train', **kwargs)


def metrics(**updates):
    return dict(dict(valid=True, auc=.4, mppe_cm=8., survival_s=10.,
                     upright_palm_rmse_first4s_m=.2, upright_intended_contact_s=0.), **updates)


def test_early_diagnostics_do_not_require_success_but_invalid_or_severe_stops():
    base = metrics()
    assert pilot.diagnostic_review(base, base, target=5_000_000)['continue_training']
    assert pilot.diagnostic_review(base, metrics(auc=.3), target=10_000_000)['continue_training']
    assert not pilot.diagnostic_review(base, metrics(auc=.3, survival_s=5.), target=5_000_000)['continue_training']
    assert not pilot.diagnostic_review(base, metrics(valid=False), target=5_000_000)['continue_training']


def test_physical_progress_is_recorded_without_automatic_extension():
    result = pilot.diagnostic_review(metrics(), metrics(survival_s=12.), target=15_000_000)
    assert result['progress'] and not result['continue_training']
    assert result['future_training_requires_separate_decision']
