import numpy as np
from real2sim2real.reconstruction_repair.elongated_shape import principal_axis,orientation_audit


def fixture():
    v=np.array([[x,y,z] for x in [-.5,.5] for y in [-.02,.02] for z in [-.02,.02]])
    axis=principal_axis(v);pose=np.eye(4);pose[2,3]=2
    k=np.array([[200.,0,64],[0,200.,64],[0,0,1]])
    mask=np.zeros((128,128),bool);mask[60:69,14:115]=True
    return axis,pose,k,mask,np.zeros_like(mask)


def test_correct_axis_and_a_quarter_turn_are_distinguished():
    a,p,k,m,h=fixture();r=orientation_audit(a,p,k,m,h)
    assert r['informative'] and r['axis_error_deg']<1
    p[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    assert orientation_audit(a,p,k,m,h)['axis_error_deg']>89


def test_axis_sign_and_axial_roll_do_not_create_false_orientation_evidence():
    a,p,k,m,h=fixture();a['axis_mesh']=(-np.array(a['axis_mesh'])).tolist()
    r=orientation_audit(a,p,k,m,h)
    assert r['axis_error_deg']<1 and not r['head_tail_distinguished'] and not r['axial_roll_observed']


def test_hidden_or_end_on_axes_are_uninformative():
    a,p,k,m,h=fixture();assert not orientation_audit(a,p,k,m,m)['informative']
    p[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    assert not orientation_audit(a,p,k,m,h)['informative']


def test_translation_error_is_reported_without_changing_the_input():
    a,p,k,m,h=fixture();p[0,3]=.2;before=p.copy();r=orientation_audit(a,p,k,m,h)
    assert r['center_error_px']>19
    np.testing.assert_array_equal(before,p)
