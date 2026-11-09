from copy import deepcopy
import pytest
from real2sim2real.controller_actor_training.initial_rest_reference import revise_row

def row():
    return dict(object_body_position=[[[0,0,1]],[[0,0,1]],[[0,0,1]]],
        object_body_wxyz=[[[1,0,0,0]]]*3,object_root_position=[[0,0,1]]*3,
        object_root_axis_angle=[[0,0,0]]*3,hand_contact_active=[[0,0,1],[0,0,1]],
        object_articulation=[0,0,0],robot_joint_positions=[[.1],[.2],[.3]],fps=30.)

def test_changes_only_initial_object_representations_and_preserves_source():
    original=row();before=deepcopy(original)
    revised,receipt=revise_row(original,[0,0,1.01,0,0,0,1])
    assert original==before
    for key in original:
        if key.startswith(('object_body_','object_root_')):
            assert revised[key][1:]==original[key][1:]
        else:assert revised[key]==original[key]
    assert revised['object_body_position'][0]==[[0,0,1.01]]
    assert receipt['first_native_seam_m']==pytest.approx(.01)

@pytest.mark.parametrize('key,value',[
    ('hand_contact_active',[[1,0,0],[0,0,0]]),('object_articulation',[0,1,0]),
    ('object_root_position',[[0,0,2]]*3),('object_root_axis_angle',[[0,0,.1]]*3)])
def test_unsupported_or_inconsistent_reference_is_rejected(key,value):
    original=row();original[key]=value
    with pytest.raises(ValueError):revise_row(original,[0,0,1.01,0,0,0,1])

def test_new_seam_and_correction_limits_are_separate():
    with pytest.raises(ValueError,match='seam'):revise_row(row(),[0,0,1.02,0,0,0,1],max_seam_m=.01)
    with pytest.raises(ValueError):revise_row(row(),[0,0,1.04,0,0,0,1])

def test_nonunit_quaternion_rejected():
    with pytest.raises(ValueError):revise_row(row(),[0,0,1,0,0,0,2])
