import numpy as np
import pytest
from real2sim2real.reconstruction_repair.fixed_scale_pose import restore_fixed_scale_projection


def test_restoring_size_preserves_projection_and_scales_translation():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    t=np.array([.1,-.2,1.4]);v=np.array([[-.2,.1,0.],[.3,-.1,.2],[.05,.1,-.3]])
    result=restore_fixed_scale_projection(R,t,.13,.277)
    before=v@R.T*.13+t;after=v@result['rotation'].T*result['scale']+result['translation']
    np.testing.assert_allclose(before[:,:2]/before[:,2:],after[:,:2]/after[:,2:],atol=1e-12)
    np.testing.assert_allclose(after[1]-after[0],(before[1]-before[0])*(.277/.13))
    np.testing.assert_array_equal(t,[.1,-.2,1.4])
    assert result['candidate_approved'] is False


def test_scale_only_override_changes_projection():
    v=np.array([[.2,.1,0.]]);t=np.array([.1,-.2,1.4])
    before=v*.13+t;wrong=v*.277+t
    assert not np.allclose(before[:,:2]/before[:,2:],wrong[:,:2]/wrong[:,2:])


@pytest.mark.parametrize('scale',[0.,-1.,np.inf,np.nan,[1.,2.,1.],[1.,1.]])
def test_invalid_or_anisotropic_scale_rejected(scale):
    with pytest.raises(ValueError):restore_fixed_scale_projection(np.eye(3),np.ones(3),scale,1.)


def test_roundoff_uniform_scale_accepted_but_reflection_rejected():
    result=restore_fixed_scale_projection(np.eye(3),np.ones(3),[.13,.130000001,.13],[.277]*3)
    assert result['scale']==.277
    with pytest.raises(ValueError):restore_fixed_scale_projection(np.diag([-1.,1.,1.]),np.ones(3),1.,2.)
