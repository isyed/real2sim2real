import numpy as np
import pytest
from real2sim2real.reconstruction.observed_seed import make,validate,failure_kind,SOURCE


def seed():
    return make(np.eye(4),[[100,0,50],[0,100,50],[0,0,1]],native_source_frame=290,
        method='verified_tracking_prefix',evidence_hashes={'trace':'a'*64})


def test_reuse_keeps_machine_tag_and_native_mapping_separate_from_description():
    value=seed();pose,camera=validate(value)
    assert value['source']==SOURCE and value['frame']==0 and value['native_source_frame']==290
    assert value['method']=='verified_tracking_prefix' and value['evidence_sha256']['trace']=='a'*64
    pose[0,3]=10
    assert value['pose_cam'][0][3]==0 and camera[0,0]==100


@pytest.mark.parametrize('bad',['source','frame','pose','camera','clean'])
def test_bad_seed_is_rejected_without_importing_a_model(tmp_path,bad):
    value=seed()
    if bad=='source':value['source']='A human-readable explanation is not a supported provenance enum'
    if bad=='frame':value['frame']=290
    if bad=='pose':value['pose_cam'][0][0]=2
    if bad=='camera':value['K'][0][0]=-1
    if bad=='clean':value['clean_trajectory_used']=True
    with pytest.raises(ValueError):validate(value)


def test_runtime_error_never_becomes_a_quality_rejection():
    assert failure_kind(1,{'error':'Unrecognized observed pose provenance'})=='runtime_failure'
    assert failure_kind(1,{'error':'No quality-valid recovery at source frame 1; reasons=[depth]'})=='quality_rejection'
    assert failure_kind(1,{'error':'No quality-valid recovery at tracker-local frame 1; reasons=[depth]'})=='quality_rejection'
    assert failure_kind(86)=='guard_stop'
    assert failure_kind(0)=='completed'
