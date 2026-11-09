import numpy as np
import pytest

from real2sim2real.reconstruction_repair.tracking_windows import bounded_windows,source_order


def test_reverse_tracking_is_ordered_by_original_source_frames():
    ids=np.arange(650,619,-1)
    assert ids[source_order(ids)].tolist()==list(range(620,651))


@pytest.mark.parametrize('ids',[[1,2,4,5,6],[-1,0,1,2,3],[1.,2.,3.,4.,5.]])
def test_window_ids_must_be_explicit_consecutive_source_indices(ids):
    with pytest.raises(ValueError):source_order(ids)


def test_windows_share_one_anchor_without_leaving_a_short_uncheckable_tail():
    for count in [5,31,32,33,34,35,64,115,601]:
        windows=list(bounded_windows(count,31))
        assert windows[0][0]==0 and windows[-1][1]==count
        assert all(end-start>=5 for start,end in windows)
        assert all(right[0]==left[1]-1 for left,right in zip(windows,windows[1:]))


def test_native_reverse_sequence_keeps_poses_and_maps_accepted_frames(tmp_path):
    import io,pickle
    import h5py,trimesh
    from PIL import Image
    from conftest import box_geometry,observations
    from real2sim2real.artifacts import digest,write_json
    from real2sim2real.reconstruction_repair.temporal_pose import TemporalPoseConfig
    from real2sim2real.reconstruction_repair.tracking_windows import refine
    g=box_geometry(9);masks=observations(g)['object_mask'];export=tmp_path/'export';export.mkdir()
    mesh=export/'object_mesh/output_aligned.glb';mesh.parent.mkdir()
    trimesh.Trimesh(g['vertices'],g['faces'],process=False).export(mesh)
    write_json(export/'wild_export.json',dict(frame_count=9,intrinsics=g['intrinsics'][0].tolist(),
        source_video_sha256='video',parent_artifact_sha256='parent'))
    for role,values in [('object_masks',masks),('human_masks',np.zeros_like(masks))]:
        path=export/role/'front_stereo_camera_left.h5';path.parent.mkdir()
        with h5py.File(path,'w') as h:h.create_dataset('frames',data=values)
    depth=tmp_path/'depth.h5'
    with h5py.File(depth,'w') as h:
        values=h.create_dataset('aligned/front_stereo_camera_left',shape=(9,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(9):
            encoded=io.BytesIO();Image.fromarray(np.full((96,128),1970,np.uint16)).save(encoded,format='PNG')
            values[i]=np.frombuffer(encoded.getvalue(),np.uint8)
    proposals=tmp_path/'poses.pkl';poses=np.tile(np.eye(4),(9,1,1));poses[:,2,3]=2.
    with proposals.open('wb') as f:pickle.dump(dict(frames=[f'{i:06d}' for i in range(9)],
        fp_poses_cam=poses[:,None],metadata=dict(proposal_only=True,gt_object_pose_available=False)),f)
    write_json(proposals.with_suffix('.proposal_receipt.json'),dict(output_sha256=digest(proposals),proposal_only=True))
    mapping=tmp_path/'mapping.json';write_json(mapping,dict(source_frame_indices=list(range(108,99,-1)),
        source_fps=30.,source_video_sha256='video',parent_artifact_sha256='parent'))
    support=tmp_path/'support.json';write_json(support,dict(ground_truth_used=False,coordinate_frame='camera',
        plane_camera=[0.,-1.,0.,.052],center_camera_m=[0.,.052,2.],radius_m=1e6))
    result=refine(export,depth,proposals,mapping,support,tmp_path/'output',
        config=TemporalPoseConfig(min_depth_points=4),window_frames=5)
    assert result['complete'] and result['accepted_source_frames']==list(range(108,99,-1))
    assert all(w['action']=='retain_already_valid_proposals' for w in result['windows'])
    assert not result['full_reference_accepted'] and not result['metric_size_certified']
    with np.load(tmp_path/'output/poses_with_acceptance.npz') as saved:
        np.testing.assert_array_equal(saved['poses_camera'],poses)
    write_json(mapping,dict(source_frame_indices=list(range(108,99,-1)),source_fps=30.,
        source_video_sha256='different-video',parent_artifact_sha256='parent'))
    with pytest.raises(ValueError,match='identities differ'):
        refine(export,depth,proposals,mapping,support,tmp_path/'wrong')
