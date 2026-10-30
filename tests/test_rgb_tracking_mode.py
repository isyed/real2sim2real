import json
from types import SimpleNamespace
import numpy as np
import pytest
from real2sim2real.reconstruction import tracking_recovery as adapter

def test_normal_tracking_calls_existing_rgb_refiner_mode(monkeypatch):
    class Tensor(np.ndarray):
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    monkeypatch.setitem(__import__('sys').modules, 'torch', SimpleNamespace(no_grad=lambda: lambda fn: fn))
    calls = []
    pose = np.eye(4).view(Tensor)
    pose[2, 3] = 3.

    def predict(**kwargs):
        calls.append(kwargs)
        return pose[None], None

    class FakePose:
        def __init__(self):
            self.mesh = self.mesh_tensors = self.glctx = None
            self.diameter = .3
            self.pose_last = pose.copy()
            self.refiner = SimpleNamespace(predict=predict)

        def prepare_registration_frame(self, K, rgb, depth, mask, *, tensor_residency):
            assert tensor_residency
            return dict(rgb_tensor=rgb, depth_tensor=depth, xyz_map_tensor=np.zeros((*depth.shape, 3)))

        def get_tf_to_centered_mesh(self):
            return np.eye(4).view(Tensor)

    upstream = SimpleNamespace(FoundationPose=FakePose)
    adapter.install_rgb_mode(upstream)
    result = upstream.FoundationPose().track_one(np.zeros((8, 8, 3)), np.ones((8, 8)), np.eye(3), 5)
    assert len(calls) == 1 and calls[0]['rgb_only'] is True and calls[0]['get_vis'] is False
    np.testing.assert_allclose(result, pose)


def test_failure_state_is_recorded_as_unaccepted(tmp_path):
    class FakePose:
        def __init__(self): self.pose_last = np.eye(4)
        def get_tf_to_centered_mesh(self): return np.eye(4)
        def track_one(self, *args, **kwargs): raise RuntimeError('No valid recovery')
    upstream = SimpleNamespace(FoundationPose=FakePose, _tensor_to_numpy=np.asarray,
                               _atomic_pickle_dump=lambda *args: None)
    adapter.install_state_recording(upstream, tmp_path/'poses.pkl')
    with pytest.raises(RuntimeError, match='No valid recovery'):
        upstream.FoundationPose().track_one()
    row = json.loads((tmp_path/'poses.states.jsonl').read_text())
    assert row['frame'] == 1 and row['accepted'] is False and 'selected_pose' not in row
    np.testing.assert_array_equal(row['previous_accepted_pose'], np.eye(4))


