import pytest
from real2sim2real.reconstruction_repair.decision import combine_decision


def test_independent_interaction_veto_overrides_geometric_success():
    primary=dict(accepted=True,reasons=[],heldout_mean_iou_after=.96)
    checks=[dict(id='hand_separation',passed=False,observed_m=.039,threshold_m=.005)]
    result=combine_decision(primary,checks)
    assert result['primary_accepted'] and not result['accepted']
    assert result['reasons']==['hand_separation']
    assert not result['full_reference_accepted'] and not result['actor_training_approved']
    result['independent_checks'][0]['observed_m']=0
    assert checks[0]['observed_m']==.039 and primary['reasons']==[]


def test_independent_pass_does_not_rescue_primary_rejection():
    result=combine_decision(dict(accepted=False,reasons=['depth_gate']),
                            [dict(id='hand_separation',passed=True)])
    assert not result['accepted'] and result['reasons']==['depth_gate']


def test_local_success_still_does_not_approve_a_full_reference():
    result=combine_decision(dict(accepted=True,reasons=[]),[dict(id='hand',passed=True)])
    assert result['accepted'] and not result['approved_for_use']


@pytest.mark.parametrize('checks',[
    [dict(id='hand')], [dict(id='hand',passed=1)],
    [dict(id='hand',passed=True),dict(id='hand',passed=False)]])
def test_unknown_or_ambiguous_independent_checks_cannot_pass(checks):
    with pytest.raises(ValueError):
        combine_decision(dict(accepted=True,reasons=[]),checks)
