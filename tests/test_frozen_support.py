import numpy as np
import pytest
import trimesh
from real2sim2real.controller_actor_training.support_check import placement_metrics


def box():
    mesh=trimesh.creation.box(extents=[1,1,1]);return [('table',mesh.vertices,mesh.faces)]


def test_surface_distance_and_inside_depth_are_distinct():
    report=placement_metrics([[0,0,.505]],box())
    assert report['passed']
    assert report['maximum_sampled_penetration_m']==0
    assert report['minimum_surface_gap_m']==pytest.approx(.005)
    buried=placement_metrics([[0,0,.47]],box())
    assert not buried['passed']
    assert buried['maximum_sampled_penetration_m']==pytest.approx(.03)
    assert not placement_metrics([[0,0,.6]],box())['passed']


def test_empty_or_nonfinite_geometry_cannot_pass():
    with pytest.raises(ValueError):placement_metrics([],box())
    with pytest.raises(ValueError):placement_metrics([[np.nan,0,0]],box())
    with pytest.raises(ValueError):placement_metrics([[0,0,0]],[])
    with pytest.raises(ValueError):placement_metrics([[0,0,0]],box(),max_gap_m=-1)


def test_open_support_cannot_be_treated_as_a_convex_solid():
    mesh=trimesh.creation.box();faces=mesh.faces[:-1]
    with pytest.raises(ValueError,match='closed convex'):
        placement_metrics([[0,0,.5]],[('open',mesh.vertices,faces)])
