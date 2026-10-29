import numpy as np
import pytest
from test_metric_seed import scene
from real2sim2real.reconstruction_repair.support_seed import propose_resting_seed


def test_resting_seed_is_bounded_and_does_not_modify_parent():
    v,f,k,t,mask,z=scene();parent=np.eye(4);parent[:3,3]=t
    seed=parent.copy();seed[1,3]-=.04;original=seed.copy()
    candidate,r=propose_resting_seed(v,f,parent,v,f,seed,k,mask,np.zeros_like(mask),z,
        np.array([0.,-1.,0.,.15]),resting_evidence='Synthetic resting cube',steps=5)
    assert r['candidate_available']
    assert np.linalg.norm(candidate[:3,3]-seed[:3,3])<=.05
    assert np.array_equal(seed,original)
    assert np.array_equal(candidate[:3,:3],seed[:3,:3])
    assert not r['actor_training_approved'] and not r['phase_propagated']


def test_support_cannot_override_image_or_search_bound():
    v,f,k,t,mask,z=scene();parent=np.eye(4);parent[:3,3]=t
    seed=parent.copy();seed[1,3]-=.4
    candidate,r=propose_resting_seed(v,f,parent,v,f,seed,k,mask,np.zeros_like(mask),z,
        np.array([0.,-1.,0.,.15]),resting_evidence='Synthetic resting cube')
    assert not r['candidate_available']
    assert np.array_equal(candidate,seed)
    assert max(x['translation_m'] for x in r['trials'])<=.05


def test_resting_phase_is_never_inferred():
    with pytest.raises(ValueError,match='Reviewed resting'):
        propose_resting_seed(*([None]*11),resting_evidence='')
