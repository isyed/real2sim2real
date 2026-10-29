import numpy as np
import pytest

from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
from real2sim2real.reconstruction_repair.metric_seed import register_translation


def scene():
    # Dense outward-facing cube, independent of any licensed object or episode.
    import itertools
    vertices = []; faces = []
    grid = np.linspace(-.25, .25, 25)
    for axis, side in itertools.product(range(3), (-1, 1)):
        start = len(vertices)
        other = [i for i in range(3) if i != axis]
        for a, b in itertools.product(grid, grid):
            p = np.zeros(3); p[axis] = side * .25; p[other] = [a, b]; vertices.append(p)
        for a, b in itertools.product(range(24), range(24)):
            i = start + a * 25 + b
            for tri in ([i, i + 25, i + 1], [i + 1, i + 25, i + 26]):
                xyz = np.asarray(vertices)[tri]
                if np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])[axis] * side < 0: tri = tri[::-1]
                faces.append(tri)
    v, f = np.asarray(vertices), np.asarray(faces)
    K = np.array([[240., 0, 160], [0, 240, 120], [0, 0, 1.]])
    t = np.array([.2, -.1, 3.])
    g = dict(vertices=v, faces=f, object_scale=np.array(1.), object_rotation=np.eye(3)[None],
             object_translation=t[None], intrinsics=K[None], world_to_camera=np.eye(4)[None])
    mask = mesh_silhouette(g, 0, 240, 320)
    depth = np.where(mask, 2.75, 0.)
    return v, f, K, t, mask, depth


def test_fresh_seed_matches_foreground_surface_not_center_depth():
    v, f, K, t, mask, depth = scene()
    pose, scores = register_translation(v, f, np.eye(3), K, mask, np.zeros_like(mask), depth)
    assert np.allclose(pose[:3, 3], t, atol=.02)
    assert scores['iou'] > .90
    assert not scores['old_translation_used']
    assert not scores['trajectory_accepted']
    assert not scores['approved_for_use']


def test_nonpenetrating_seed_is_not_silently_reported_as_resting_contact():
    from real2sim2real.reconstruction_repair.metric_seed import assess_seed
    v,f,K,t,mask,depth=scene();p=np.eye(4);p[:3,3]=t
    plane=np.array([0.,-1.,0.,.15])
    grounded=assess_seed(v,f,p,K,mask,np.zeros_like(mask),depth,plane)
    assert grounded['resting_clearance_within_2cm']
    original=p.copy();p[1,3]-=.05
    floating=assess_seed(v,f,p,K,mask,np.zeros_like(mask),depth,plane)
    assert floating['nonpenetration_within_2cm'] and not floating['resting_clearance_within_2cm']
    assert floating['support_gap_m']==pytest.approx(.05)
    assert np.array_equal(p[:3,:3],original[:3,:3]) and not floating['poses_modified']
    with pytest.raises(ValueError):assess_seed(v,f,p,K,mask,np.zeros_like(mask),depth,plane*2)


def test_parent_comparison_recomputes_both_on_same_pixels_and_rejects_harmful_seed():
    from real2sim2real.reconstruction_repair.metric_seed import compare_seeds
    v,f,K,t,mask,depth=scene();parent=np.eye(4);parent[:3,3]=t
    plane=np.array([0.,-1.,0.,.15]);proposal=parent.copy();proposal[0,3]+=.08
    result=compare_seeds(v,f,parent,v,f,proposal,K,mask,np.zeros_like(mask),depth,plane,require_resting=True)
    assert not result['proposal_passed'] and 'previous_good_image_regressed' in result['reasons']
    assert result['same_pixel_evidence'] and not result['full_reference_accepted']
    same=compare_seeds(v,f,parent,v,f,parent,K,mask,np.zeros_like(mask),depth,plane,require_resting=True)
    assert same['proposal_passed'] and not same['actor_training_approved']


def test_occlusion_and_invalid_depth_cannot_create_a_seed():
    v, f, K, t, mask, depth = scene()
    with pytest.raises(ValueError, match='100 visible'):
        register_translation(v, f, np.eye(3), K, mask, mask, depth)
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid'):
        register_translation(v, f, np.eye(3), K, mask, np.zeros_like(mask), depth)


def test_seed_boundary_verifies_calibration_and_preserves_parent(tmp_path):
    import json
    from real2sim2real.reconstruction_repair.bundle import arrays_atomic, file_sha256
    from real2sim2real.reconstruction_repair.metric_seed import seed_from_calibration
    v, f, K, t, mask, depth = scene()
    calibration = tmp_path / 'calibration'; calibration.mkdir()
    arrays_atomic(calibration / 'calibrated_mesh.npz', dict(vertices=v, faces=f))
    report = calibration / 'calibration.json'
    report.write_text(json.dumps(dict(candidate_ready=True, requires_retracking=True,
        old_poses_valid_for_resized_mesh=False, mesh_sha256=file_sha256(calibration / 'calibrated_mesh.npz'))))
    evidence = tmp_path / 'evidence.npz'
    arrays_atomic(evidence, dict(rotation=np.eye(3), intrinsics=K, object_mask=mask,
        occluder_mask=np.zeros_like(mask), depth_m=depth))
    provenance = tmp_path / 'source.json'
    spec = dict(schema='r2s2r_engines.metric-seed-evidence/1', source='synthetic predicted depth',
        clean_trajectory_used=False, input_sha256=file_sha256(evidence), calibration_sha256=file_sha256(report))
    provenance.write_text(json.dumps(spec)); before = file_sha256(report)
    result = seed_from_calibration(calibration, evidence, provenance, tmp_path / 'seed')
    assert result['iou'] > .9
    assert file_sha256(report) == before
    assert not (tmp_path / 'seed/manifest.json').exists()
    spec['clean_trajectory_used'] = True; provenance.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='verified'):
        seed_from_calibration(calibration, evidence, provenance, tmp_path / 'invalid')


def test_rgb_rotation_preserves_source_projection_under_mesh_frame_change():
    from scipy.spatial.transform import Rotation
    from real2sim2real.reconstruction_repair.metric_seed import rotation_from_mesh_prediction
    original = Rotation.from_euler('xyz', [.2, -.5, .8]).as_matrix()
    frame = Rotation.from_euler('xyz', [.3, .7, -.2]).as_matrix()
    q = Rotation.from_matrix(original).as_quat()[[3, 0, 1, 2]]
    transform = np.eye(4); transform[:3, :3] = frame
    aligned = rotation_from_mesh_prediction(q, transform)
    assert np.allclose(aligned @ frame, original)
    with pytest.raises(ValueError, match='unit wxyz'):
        rotation_from_mesh_prediction(q * 2, transform)
    transform[0, 0] *= 2
    with pytest.raises(ValueError, match='rigid'):
        rotation_from_mesh_prediction(q, transform)
