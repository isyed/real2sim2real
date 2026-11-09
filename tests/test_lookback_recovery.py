import numpy as np
import pytest
import trimesh
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
from real2sim2real.reconstruction_repair.local_registration import LocalRegistrationConfig
from real2sim2real.reconstruction_repair.lookback_recovery import propose_pair


def fixture():
    mesh=trimesh.creation.box(extents=[.4,.4,.2])
    for _ in range(3):mesh=mesh.subdivide()
    camera=np.array([[200.,0,64],[0,200,64],[0,0,1]])
    previous=np.eye(4);previous[2,3]=2.04;anchor=previous.copy();anchor[2,3]=2.05
    geometry=dict(vertices=mesh.vertices,faces=mesh.faces,object_scale=np.array(1.),
        object_rotation=np.tile(np.eye(3),(2,1,1)),object_translation=np.array([[0,0,2.03],[0,0,2.01]]),
        intrinsics=np.tile(camera,(2,1,1)),world_to_camera=np.tile(np.eye(4),(2,1,1)))
    objects=np.stack([mesh_silhouette(geometry,i,128,128) for i in [0,1]])
    support=dict(plane_camera=[0,0,1,0],center_camera_m=[0,0,0],radius_m=10.,
        evidence='Reviewed static support in source camera',ground_truth_used=False)
    return mesh,previous,anchor,camera,objects,np.zeros_like(objects),np.full(objects.shape,1.9),support


def call(data,**kw):
    mesh,*args=data
    return propose_pair(mesh.vertices,mesh.faces,*args,source_frame_ids=[10,11],
        config=LocalRegistrationConfig(max_evaluations=1),**kw)


def test_both_native_frames_and_joins_checked_without_mutating_anchor():
    data=fixture();before=data[2].copy();poses,report=call(data,ray_shifts_m=[-.02])
    assert poses is not None and report['all_cpu_gates_passed']
    assert np.array_equal(data[2],before) and report['prior_prefix_before_anchor_preserved']
    assert report['gpu_revalidation_required'] and not report['full_reference_accepted']
    assert report['gpu_revalidation_source_frame_ids']==[10,11]
    assert report['continuation_seed_source_frame_after_both_gpu_checks']==11
    assert not report['thresholds_relaxed'] and poses.shape==(2,4,4)


def test_valid_next_frame_cannot_hide_a_bad_revised_anchor():
    data=list(fixture());data[4]=np.roll(data[4],50,axis=2)
    poses,report=call(data,ray_shifts_m=[-.02])
    assert poses is None and report['rows'][0]['anchor_rejected_for']


def test_missing_next_image_never_becomes_a_carried_pose():
    data=list(fixture());data[4][1]=False
    poses,report=call(data,ray_shifts_m=[-.02])
    assert poses is None and 'next_frame_error' in report['rows'][0]


def test_anchor_motion_limit_is_not_relaxed_to_make_the_pair_pass():
    data=list(fixture());data[1][0,3]=1.
    poses,report=call(data,ray_shifts_m=[-.02])
    assert poses is None and 'translation_step' in report['rows'][0]['anchor_rejected_for']
    with pytest.raises(ValueError):call(fixture(),ray_shifts_m=[-.2])
