import numpy as np
import pytest
from real2sim2real.reconstruction_repair.depth_scale import static_floor_gain,metric_pointmap


def test_metric_pointmap_preserves_native_camera_and_explicit_metre_depth():
    k=np.array([[40,0,15],[0,40,10],[0,0,1.]])
    z=np.full((20,30),2.7,np.float32);p=metric_pointmap(z,k)
    assert np.allclose(p[10,15],[0,0,2.7])
    uv=p@k.T;uv=uv[...,:2]/uv[...,2:]
    v,u=np.indices(z.shape);assert np.allclose(uv,np.stack((u,v),axis=-1),atol=1e-5)
    with pytest.raises(ValueError,match='units'):metric_pointmap(z*1000,k)


def samples(gain):
    rng=np.random.default_rng(0)
    p=np.c_[rng.uniform(-1,1,600),np.full(600,1.),rng.uniform(2,4,600)]/gain
    return p,np.repeat([0,1],300),np.array([0.,-1.,0.,1.])


def test_static_floor_removes_known_scale_drift_without_metric_truth_claim():
    p,r,plane=samples(1.035);gain,report=static_floor_gain(p,r,plane)
    assert gain==pytest.approx(1.035)
    assert np.max(abs((p*gain)@plane[:3]+plane[3]))<1e-10
    assert report['anchor_is_predicted'] and not report['absolute_metric_accuracy_established']


@pytest.mark.parametrize('case',['regions','inconsistent','large_drift','invalid_plane','nan'])
def test_invalid_or_nonstatic_floor_evidence_is_not_silently_rescaled(case):
    p,r,plane=samples(1.02)
    if case=='regions':r[:]=0
    if case=='inconsistent':p[r==1]*=1.08
    if case=='large_drift':p*=.7
    if case=='invalid_plane':plane*=2
    if case=='nan':p[5,1]=np.nan
    with pytest.raises(ValueError):static_floor_gain(p,r,plane)
