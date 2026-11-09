import numpy as np
import trimesh
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
from real2sim2real.reconstruction_repair.observation_preservation import preserve


def fixture():
    mesh=trimesh.creation.box(extents=[.4,.4,.2])
    for _ in range(4):mesh=mesh.subdivide()
    k=np.array([[200.,0,64],[0,200,64],[0,0,1]])
    target=np.tile(np.eye(4),(5,1,1));target[:,2,3]=2.
    geometry=dict(vertices=mesh.vertices,faces=mesh.faces,object_scale=np.array(1.),object_rotation=target[:,:3,:3],
        object_translation=target[:,:3,3],intrinsics=np.tile(k,(5,1,1)),world_to_camera=np.tile(np.eye(4),(5,1,1)))
    objects=np.stack([mesh_silhouette(geometry,i,128,128) for i in range(5)])
    original=target.copy();original[3,0,3]=.12
    candidate=target.copy();candidate[2,0,3]=.02
    data=dict(vertices=mesh.vertices,faces=mesh.faces,poses_camera=original,frame_ids=np.arange(5),timestamps_s=np.arange(5)/30,
        intrinsics=np.tile(k,(5,1,1)),object_mask=objects,occluder_mask=np.zeros_like(objects),depth_m=np.full(objects.shape,1.9),
        fit=np.array([1,0,1,0,1],bool),locked=np.array([1,0,0,0,1],bool),support_plane_camera=np.tile([0.,0,1,0],(5,1)),
        support_center_camera=np.zeros((5,3)),support_radius_m=np.full(5,10.))
    return data,candidate


def test_good_original_frame_is_preserved_without_losing_other_improvements():
    data,candidate=fixture();before=data['poses_camera'].copy();proposed=candidate.copy()
    hybrid,report=preserve(data,candidate)
    assert report['restored_source_frame_ids']==[2] and report['decision']['accepted']
    np.testing.assert_array_equal(hybrid[2],before[2]);np.testing.assert_array_equal(hybrid[3],proposed[3])
    np.testing.assert_array_equal(data['poses_camera'],before);np.testing.assert_array_equal(candidate,proposed)
    assert not report['thresholds_relaxed'] and report['requires_gpu_validation']


def test_good_image_with_invalid_depth_is_never_restored():
    data,candidate=fixture();data['poses_camera'][2,2,3]=2.25
    geometry=dict(vertices=data['vertices'],faces=data['faces'],object_scale=np.array(1.),
        object_rotation=data['poses_camera'][:,:3,:3],object_translation=data['poses_camera'][:,:3,3],
        intrinsics=data['intrinsics'],world_to_camera=np.tile(np.eye(4),(5,1,1)))
    data['object_mask'][2]=mesh_silhouette(geometry,2,128,128);candidate[2]=np.eye(4);candidate[2,2,3]=2.
    hybrid,report=preserve(data,candidate)
    assert 2 not in report['restored_source_frame_ids']
    assert not report['decision']['accepted']


def test_preservation_cannot_hide_a_broken_locked_anchor():
    data,candidate=fixture();candidate[0,0,3]=.02
    _,report=preserve(data,candidate)
    assert 'locked_anchor_changed' in report['decision']['reasons']


def test_preservation_keeps_global_motion_checks_active():
    data,candidate=fixture();candidate[3,0,3]=.3
    _,report=preserve(data,candidate)
    assert report['all_window_and_motion_gates_rechecked']
    assert 'motion_step_gate' in report['decision']['reasons']


def test_heldout_evidence_is_not_used_to_choose_restored_poses():
    data,candidate=fixture();candidate[1,0,3]=.02
    _,report=preserve(data,candidate)
    assert 1 not in report['restored_source_frame_ids']
    assert not report['heldout_frames_used_for_restoration']
    assert 'previous_good_image_regressed' in report['decision']['reasons']
