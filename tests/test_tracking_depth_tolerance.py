import numpy as np
import pytest

from real2sim2real.reconstruction.recovery_quality import configured_limits, protocol, rejection_reasons


def test_explicit_depth_variant_changes_only_depth_acceptance():
    pose=np.eye(4)
    metrics=dict(iou=.7,precision=.8,recall=.8)
    geometry=dict(depth_points=200,depth_error_m=.11,support_penetration_m=0.,behind_camera=False)
    assert rejection_reasons(pose,pose,metrics,geometry,recovery=True)==['depth']
    assert rejection_reasons(pose,pose,metrics,geometry,recovery=True,
        limits=configured_limits(depth_error_m=.12))==[]
    geometry['support_penetration_m']=.03
    assert rejection_reasons(pose,pose,metrics,geometry,recovery=True,
        limits=configured_limits(depth_error_m=.12))==['support']


def test_protocol_retains_explicit_tolerance_and_all_other_limits():
    default=protocol()['limits'];variant=protocol(depth_error_m=.12)['limits']
    assert default['depth_error_m']==.1 and variant['depth_error_m']==.12
    assert {k:v for k,v in default.items() if k!='depth_error_m'}=={k:v for k,v in variant.items() if k!='depth_error_m'}


@pytest.mark.parametrize('value',[True,0.,.21,float('nan'),float('inf')])
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError):configured_limits(depth_error_m=value)
