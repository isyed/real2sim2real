import json
import numpy as np
import pytest

from real2sim2real.reconstruction_repair.bundle import arrays_atomic, file_sha256
from real2sim2real.reconstruction_repair.metric_calibration import CalibrationConfig, calibrate_object, diagnose, diameter


def evidence_for(bundle):
    n=len(bundle.geometry['frame_ids'])
    evidence=dict(frame_ids=bundle.geometry['frame_ids'].copy(),
        depth_m=np.full(bundle.observations['object_mask'].shape,4.),
        fit=np.arange(n)%2==0, floor_planes=np.tile([0.,-1.,0.,1.5],(n,1)),
        floor_valid=np.ones(n,bool))
    provenance=dict(schema='r2s2r_engines.metric-calibration-evidence/1',source='synthetic depth',
        floor_roi_source='synthetic floor',clean_trajectory_used=False,
        input_manifest_sha256=file_sha256(bundle.root/'manifest.json'))
    return evidence,provenance


def test_scale_hypothesis_never_reuses_pose(make_bundle,tmp_path):
    b=make_bundle(); e,p=evidence_for(b)
    human_hash=file_sha256(b.root/'human.npz')
    report=diagnose(b,e,p)
    assert report['candidate_ready']
    assert report['scale_multiplier']==pytest.approx(2.,rel=.05)
    assert not report['old_poses_valid_for_resized_mesh']
    assert not report['approved_for_use']
    ep=tmp_path/'depth.npz'; arrays_atomic(ep,e);p['evidence_sha256']=file_sha256(ep)
    pp=tmp_path/'provenance.json';pp.write_text(json.dumps(p))
    out=tmp_path/'calibration';calibrate_object(b.root,ep,pp,out)
    assert (out/'calibrated_mesh.npz').exists()
    assert not (out/'manifest.json').exists()
    with np.load(out/'calibrated_mesh.npz') as mesh:
        assert set(mesh.files)=={'vertices','faces'}
    assert file_sha256(b.root/'human.npz')==human_hash


def test_holdout_can_veto_but_not_fit_scale(make_bundle):
    b=make_bundle();e,p=evidence_for(b);first=diagnose(b,e,p)
    e['depth_m'][~e['fit']]*=2
    rejected=diagnose(b,e,p)
    assert rejected['scale_multiplier']==first['scale_multiplier']
    assert 'heldout_extent_disagreement' in rejected['reasons']
    assert not rejected['candidate_ready']


@pytest.mark.parametrize('failure',['missing_split','bad_hash','clean','nan','no_depth'])
def test_invalid_evidence_is_rejected(make_bundle,failure):
    b=make_bundle();e,p=evidence_for(b)
    if failure=='missing_split':e['fit'][:]=True
    if failure=='bad_hash':p['input_manifest_sha256']='wrong'
    if failure=='clean':p['clean_trajectory_used']=True
    if failure=='nan':e['depth_m'][0,0,0]=np.nan
    if failure=='no_depth':e['depth_m'][0]=0
    with pytest.raises(ValueError):diagnose(b,e,p)


@pytest.mark.parametrize('failure',['height','normal','floor_fit','scale'])
def test_inconsistent_geometry_is_rejected(make_bundle,failure):
    b=make_bundle();e,p=evidence_for(b)
    if failure=='height':e['floor_planes'][-1,3]*=1.3
    if failure=='normal':e['floor_planes'][-1,:3]=[0,0,-1]
    if failure=='floor_fit':e['floor_valid'][-1]=False
    if failure=='scale':e['depth_m']*=5
    assert not diagnose(b,e,p)['candidate_ready']


def test_extent_is_orientation_invariant():
    from scipy.spatial.transform import Rotation
    points=np.array([[0.,0,0],[1,0,0],[1,2,0],[0,2,0]])
    assert diameter(points)==pytest.approx(np.sqrt(5))
    assert diameter(points@Rotation.from_euler('xyz',[.4,.6,.8]).as_matrix())==pytest.approx(np.sqrt(5))
    assert diameter(np.array([[0.,0],[1,0],[2,0]]))==pytest.approx(2)


def test_frame_counts_are_not_silently_truncated():
    with pytest.raises(ValueError):CalibrationConfig(min_split_frames=3.5).validate()
