import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.reconstruction.recovery_quality import (Limits, consistency, local_proposals,
    rejection_reasons, rigid, sample_vertices)


def good_geometry():
    return dict(depth_points=200, depth_error_m=.03, behind_camera=False,
                support_penetration_m=0., depth_shift_m=0., centroid_shift_px=[0., 0.])


def test_bad_depth_and_support_are_detected_despite_good_silhouette():
    geometry = good_geometry(); geometry.update(depth_error_m=.2, support_penetration_m=.05)
    reasons = rejection_reasons(np.eye(4), np.eye(4), dict(iou=.9, precision=1., recall=.9), geometry)
    assert reasons == ['depth', 'support']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1])
def test_invalid_depth_does_not_pass(bad):
    geometry = good_geometry(); geometry['depth_error_m'] = bad
    assert 'depth' in rejection_reasons(np.eye(4), np.eye(4), dict(iou=.9, precision=.9, recall=.9), geometry)


def test_rotation_and_translation_jumps_require_recovery():
    pose = np.eye(4); pose[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix(); pose[0, 3] = .15
    assert rejection_reasons(pose, np.eye(4), dict(iou=.9, precision=.9, recall=.9), good_geometry()) == ['translation_step', 'rotation_step']


def test_recovery_cannot_lower_silhouette_threshold():
    metrics = dict(iou=.57, precision=.9, recall=.9)
    assert not rejection_reasons(np.eye(4), np.eye(4), metrics, good_geometry())
    assert rejection_reasons(np.eye(4), np.eye(4), metrics, good_geometry(), recovery=True) == ['iou']


def scene():
    x, y = np.meshgrid(np.linspace(-.2, .2, 30), np.linspace(-.2, .2, 30))
    vertices = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    pose = np.eye(4); pose[2, 3] = 2.
    K = np.array([[100., 0, 32.], [0, 100., 32.], [0, 0, 1.]])
    depth = np.full((64, 64), 2.)
    observed = np.ones((64, 64), bool)
    table = dict(plane_camera=[0., -1., 0., .4], center_camera_m=[0., .4, 2.], radius_m=.5)
    return vertices, pose, K, depth, observed, table


def test_depth_consistency_does_not_treat_translation_as_surface_depth():
    vertices, pose, K, depth, observed, table = scene()
    measured = consistency(vertices, pose, K, depth, observed, table)
    assert measured['depth_points'] > 100
    assert measured['depth_error_m'] == pytest.approx(0.)
    pose[2, 3] += .2
    moved = consistency(vertices, pose, K, depth, observed, table)
    assert moved['depth_error_m'] == pytest.approx(.2)
    assert moved['depth_shift_m'] == pytest.approx(-.2)


def test_table_check_allows_lifting_and_ignores_below_plane_points_outside_footprint():
    vertices, pose, K, depth, observed, table = scene()
    lifted = consistency(vertices, pose, K, depth, observed, table)
    assert lifted['support_penetration_m'] == 0
    pose[1, 3] = .5
    penetrating = consistency(vertices, pose, K, depth, observed, table)
    assert penetrating['support_penetration_m'] > .02
    pose[0, 3] = 2.
    outside = consistency(vertices, pose, K, depth, observed, table)
    assert outside['support_penetration_m'] == 0


def test_recovery_proposals_are_rigid_bounded_and_do_not_snap_to_table():
    previous = np.eye(4); previous[2, 3] = 3.
    geometry = good_geometry(); geometry.update(depth_shift_m=-10., centroid_shift_px=[500., -500.])
    K = np.diag([100., 100., 1.])
    poses = local_proposals(previous, previous, K, geometry)
    assert all(rigid(p) for p in poses)
    assert max(np.linalg.norm(p[:3, 3]-previous[:3, 3]) for p in poses) <= .08000001
    # Missing observations produce only proposals to be independently checked.
    geometry['depth_shift_m'] = None
    assert len(local_proposals(previous, previous, K, geometry)) == 2


def test_sampler_preserves_mesh_bounds():
    vertices = np.random.default_rng(6).normal(size=(1000, 3))*.02
    sampled = sample_vertices(vertices)
    np.testing.assert_array_equal(sampled.min(0), vertices.min(0))
    np.testing.assert_array_equal(sampled.max(0), vertices.max(0))


@pytest.mark.parametrize('gpu_accepts',[True,False])
def test_continuous_cpu_hypothesis_requires_independent_gpu_revalidation(monkeypatch,tmp_path,gpu_accepts):
    class Tensor(np.ndarray):
        def numpy(self):return np.asarray(self)
        def cpu(self):return self
    torch=SimpleNamespace(as_tensor=lambda x,**kw:np.asarray(x,dtype=kw.get('dtype')).view(Tensor),
        eye=lambda n:np.eye(n,dtype=np.float32).view(Tensor),float32=np.float32,no_grad=lambda:lambda f:f)
    monkeypatch.setitem(__import__('sys').modules,'torch',torch)
    from real2sim2real.reconstruction import tracking_recovery as adapter
    vertices,initial,K,depth,observed,table=scene();registered=[];calls=[]
    class FakePose:
        def __init__(self):
            self.mesh_ori=SimpleNamespace(vertices=vertices,faces=np.array([[0,1,2]]))
            self.pose_last=torch.as_tensor(initial,dtype=np.float32)
        def get_tf_to_centered_mesh(self):return torch.eye(4)
        def track_one(self,*a,**kw):
            p=initial.copy();p[0,3]=.15;return p
        def register(self,**kw):
            registered.append(True)
            raise RuntimeError('Expected global fallback after GPU rejection')
    def iterator(*a,**kw):
        for frame in range(2):yield str(frame),np.zeros((64,64,3)),depth,np.zeros_like(observed),observed,K
    def render(model,candidates,K,shape):
        return np.array([.9 if gpu_accepts and abs(p[0,3]-.025)<1e-6 else .4 for p in candidates])
    def continuous(*args):
        calls.append(True);p=initial.copy();p[0,3]=.025
        return dict(pose_camera=p.tolist(),proposal_valid_for_tracking=True)
    upstream=SimpleNamespace(FoundationPose=FakePose,_foundationpose_frame_iterator=iterator,
        _atomic_pickle_dump=lambda *a:None,_tensor_to_numpy=lambda value:value.numpy(),
        _render_candidate_masks=render,measure_visible_silhouettes=lambda a,*rest:a,
        _selected_metric_values=lambda measured,i:dict(iou=float(measured[i]),precision=.9,recall=.9))
    adapter.install_recovery(upstream,table,tmp_path/'poses.pkl',continuous_recovery=continuous)
    model=upstream.FoundationPose();frames=upstream._foundationpose_frame_iterator();next(frames);frame=next(frames)
    if gpu_accepts:
        result=model.track_one(frame[1],frame[2],frame[-1],5)
        assert result[0,3]==pytest.approx(.025)
        event=json.loads((tmp_path/'poses.recovery.jsonl').read_text())
        assert event['method']=='continuous_cpu_proposal_gpu_verified'
        assert not event['continuous_cpu_recovery']['gpu_revalidation'][0]['rejected_for']
        assert not registered
    else:
        with pytest.raises(RuntimeError,match='global fallback'):model.track_one(frame[1],frame[2],frame[-1],5)
        assert registered
        np.testing.assert_allclose(model.pose_last,initial)
    assert len(calls)==1 and model.continuous_calls==1


def test_adapter_rolls_back_bad_update_and_records_actual_recovery(monkeypatch, tmp_path):
    import real2sim2real.reconstruction.recovery_quality as quality
    # CPU stand-in for the tensor operations at the adapter boundary; actual
    # FoundationPose/CUDA execution belongs to the guarded integration trial.
    class Tensor(np.ndarray):
        def clone(self): return self.copy()
        def numpy(self): return np.asarray(self)
        def cpu(self): return self
    torch = SimpleNamespace(
        as_tensor=lambda x, **kw: np.asarray(x, dtype=kw.get('dtype')).view(Tensor),
        eye=lambda n: np.eye(n, dtype=np.float32).view(Tensor),
        float32=np.float32, no_grad=lambda: lambda function: function)
    monkeypatch.setitem(__import__('sys').modules, 'torch', torch)
    from real2sim2real.reconstruction import tracking_recovery as adapter
    vertices, initial, K, depth, observed, table = scene()
    tensor = torch.as_tensor

    class FakePose:
        def __init__(self):
            self.mesh_ori = SimpleNamespace(vertices=vertices)
            self.pose_last = tensor(initial, dtype=torch.float32)

        def get_tf_to_centered_mesh(self):
            return torch.eye(4)

        def track_one(self, *args, **kwargs):
            self.pose_last = self.pose_last.clone(); self.pose_last[0, 3] += .15
            return self.pose_last.numpy()

        def register(self, **kwargs):
            raise AssertionError('Local valid recovery should avoid global registration')

    def iterator(*args, **kwargs):
        for frame in range(2):
            yield f'{frame:06d}', np.zeros((64, 64, 3)), depth, np.zeros_like(observed), observed, K
    upstream = SimpleNamespace(FoundationPose=FakePose,
        _foundationpose_frame_iterator=iterator, _atomic_pickle_dump=lambda *a: None,
        _tensor_to_numpy=lambda value: value.numpy(),
        _render_candidate_masks=lambda self, candidates, K, shape: np.ones((len(candidates), *shape), bool),
        measure_visible_silhouettes=lambda *a: None,
        _selected_metric_values=lambda *a: dict(iou=.9, precision=.9, recall=.9))
    adapter.install_recovery(upstream, table, tmp_path/'poses.pkl')
    model = upstream.FoundationPose()
    frames = upstream._foundationpose_frame_iterator()
    next(frames); frame = next(frames)
    result = model.track_one(frame[1], frame[2], frame[-1], 5)
    np.testing.assert_allclose(result, initial)
    np.testing.assert_allclose(model.pose_last.numpy(), initial)
    event = json.loads((tmp_path/'poses.recovery.jsonl').read_text())
    assert event['frame'] == 1 and event['recovery']
    assert 'translation_step' in event['original']['rejected_for']
    assert not event['selected']['rejected_for']
    assert event['proposed_translation_step_m'] > .08
    assert len(event['local_reports']) == len(event['local_poses_camera'])


def test_float32_local_proposals_remain_inside_unchanged_translation_gate():
    rng = np.random.default_rng(83)
    K = np.diag([800., 800., 1.])
    metrics = dict(iou=.9, precision=.9, recall=.9)
    for _ in range(500):
        previous = np.eye(4, dtype=np.float32)
        previous[:3, 3] = [*rng.uniform(-2., 2., 2), rng.uniform(1., 8.)]
        geometry = good_geometry()
        geometry.update(depth_shift_m=float(rng.choice([-1., 1.])),
                        centroid_shift_px=rng.uniform(-100., 100., 2))
        candidates = local_proposals(previous, previous, K, geometry).astype(np.float32)
        for candidate in candidates:
            # Exactly the dtype used by the adapter, without tolerance inflation.
            assert 'translation_step' not in rejection_reasons(
                candidate, previous, metrics, geometry, recovery=True)
