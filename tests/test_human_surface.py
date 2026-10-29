import numpy as np
import pytest
from real2sim2real.human_surface import native_to_world


def test_native_surface_units_axes_and_world_transform():
    native=np.array([[[100.,-200.,-300.],[0.,0.,-100.]]])
    transform=np.eye(4)
    transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    transform[:3,3]=[2,3,4]
    result=native_to_world(native,transform)
    np.testing.assert_allclose(result,[[[0,4,7],[2,3,5]]])
    np.testing.assert_array_equal(native,[[[100,-200,-300],[0,0,-100]]])


def test_reject_invalid_surface_or_nonrigid_transform():
    with pytest.raises(ValueError):native_to_world(np.full((1,3,3),np.nan),np.eye(4))
    with pytest.raises(ValueError):native_to_world(np.zeros((3,3)),np.eye(4))
    transform=np.eye(4);transform[0,0]=2
    with pytest.raises(ValueError):native_to_world(np.zeros((1,3,3)),transform)
