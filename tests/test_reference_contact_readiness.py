import numpy as np
import pytest
from real2sim2real.retargeting.reference_contact import audit


def inputs():
    return (np.array([[0.,0,0],[2,0,0],[0,2,0]]),np.array([[0,1,2]]),
            np.eye(4)[None],np.array([[[[.5,.5,.005]],[[3.,3.,3.]]]]))


def test_face_contact_is_found_even_when_all_vertices_are_far():
    v,f,t,p=inputs();before=p.copy();r=audit(v,f,t,p)
    assert r['hands'][0]['minimum_vertex_distance_m']>.5
    assert r['hands'][0]['frames_within_threshold']==1
    assert r['hands'][1]['frames_within_threshold']==0
    assert not r['labels_modified'] and not r['physical_contact_established']
    np.testing.assert_array_equal(p,before)


def test_common_rigid_transform_preserves_proximity():
    v,f,t,p=inputs();a=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    shift=np.array([3.,-2.,1.]);t[0,:3,:3]=a;t[0,:3,3]=shift;p=p@a.T+shift
    r=audit(v,f,t,p)
    assert [x['frames_within_threshold'] for x in r['hands']]==[1,0]


def test_bad_inputs_and_search_overflow_fail_without_inventing_labels():
    v,f,t,p=inputs()
    with pytest.raises(ValueError):audit(v,f,t,p,threshold_m=0)
    with pytest.raises(ValueError):audit(v,f,t,p*np.nan)
    with pytest.raises(ValueError):audit(v,np.repeat(f,3,axis=0),t,p,maximum_candidates_per_point=1)
