import copy
import pytest
from real2sim2real.controller_actor_training.object_stability import assess_settling


def summary():
    return dict(finite=True,backend='MuJoCo-C CPU',robot_present=False,
        max_displacement_m=.03,max_rotation_deg=2.,settling_tail=dict(displacement_m=.001,rotation_deg=.1))


def test_cpu_pass_does_not_certify_cloud_or_training():
    result=assess_settling(summary())
    assert result['passed'] and result['matching_gpu_confirmation_required']
    assert not result['training_ready_established'] and not result['task_success_established']


def test_final_stillness_does_not_hide_large_initial_ejection():
    value=summary();value['max_displacement_m']=2.77;value['settling_tail']['displacement_m']=0.
    result=assess_settling(value)
    assert not result['passed'] and 'max_displacement_m:above_limit' in result['failed_checks']


@pytest.mark.parametrize('field',['max_rotation_deg','max_displacement_m'])
def test_missing_nonfinite_measurements_fail_closed(field):
    value=summary();value[field]=float('nan')
    assert not assess_settling(value)['passed']
    value.pop(field)
    assert not assess_settling(value)['passed']


def test_limits_explicit_and_no_input_mutation():
    value=summary();before=copy.deepcopy(value)
    assert not assess_settling(value,max_displacement_m=.02)['passed']
    assert value==before
    with pytest.raises(ValueError):assess_settling(value,max_rotation_deg=-1)
