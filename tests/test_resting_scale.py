import numpy as np
import pytest
from real2sim2real.reconstruction_repair.resting_scale import propose


def inputs():
    v=np.array([[-.1,-.1,-.1],[-.1,.1,.1],[.1,-.1,.1],[.1,.1,-.1]])
    p=np.tile(np.eye(4),(6,1,1));p[:,:3,3]=[0.,1./.95-.1,2.]
    return v,p,np.array([0.,-1.,0.,1.]),np.arange(6),np.arange(6)%2==0


def test_ground_hypothesis_preserves_projection_and_input_geometry():
    v,p,plane,ids,fit=inputs();old=p.copy()
    mesh,poses,r=propose(v,p,plane,ids,fit,resting_evidence='Reviewed stationary wheels resting on floor')
    assert r['accepted'] and r['scale_multiplier']==pytest.approx(.95)
    for before,after in zip(p,poses):
        a=v@before[:3,:3].T+before[:3,3];b=mesh@after[:3,:3].T+after[:3,3]
        np.testing.assert_allclose(a[:,:2]/a[:,2:],b[:,:2]/b[:,2:],atol=1e-15)
    np.testing.assert_array_equal(p,old)
    assert not r['depth_model_validated'] and not r['complete_reference_approved']


def test_heldout_ground_disagreement_stays_rejected():
    v,p,plane,ids,fit=inputs();p[~fit,1,3]+=.08
    _,_,r=propose(v,p,plane,ids,fit,resting_evidence='Reviewed ground phase')
    assert not r['accepted'] and 'heldout_ground_disagreement' in r['reasons']


def test_float32_inputs_remain_unchanged_with_float64_proposals():
    v,p,plane,ids,fit=inputs();p=p.astype(np.float32);before=p.copy()
    mesh,poses,r=propose(v.astype(np.float32),p,plane,ids,fit,resting_evidence='Reviewed floor contact')
    np.testing.assert_array_equal(p,before)
    assert r['accepted'] and poses.dtype==np.float64 and mesh.dtype==np.float64


def test_scale_is_not_clipped_to_make_a_large_change_pass():
    v,p,plane,ids,fit=inputs();p[:,1,3]=1.5
    _,_,r=propose(v,p,plane,ids,fit,resting_evidence='Reviewed ground phase')
    assert not r['accepted'] and 'scale_bound' in r['reasons']


@pytest.mark.parametrize('kind',['phase','split','floor','timeline'])
def test_missing_or_invalid_evidence_is_rejected(kind):
    v,p,plane,ids,fit=inputs();text='Reviewed floor contact'
    if kind=='phase':text=''
    if kind=='split':fit[:]=True
    if kind=='floor':plane*=2
    if kind=='timeline':ids[2]=ids[1]
    with pytest.raises(ValueError):propose(v,p,plane,ids,fit,resting_evidence=text)
