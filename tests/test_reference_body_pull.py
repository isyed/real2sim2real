import numpy as np
import pytest
wp = pytest.importorskip("warp", reason="Requires the optional simulation runtime")
pytest.importorskip("newton", reason="Requires the optional simulation runtime")
from flash_chord.runtime.actions.sonic import SonicJointResidualActionConfig,pull_reference_body_targets


def test_default_prior_is_unchanged_and_invalid_controller_bounds_are_rejected():
    assert SonicJointResidualActionConfig().reference_pull_gain==0
    for kw in [dict(reference_pull_gain=-.1),dict(reference_pull_gain=1.1),dict(reference_pull_gain=float('nan')),dict(reference_pull_max_rad=.4)]:
        with pytest.raises(ValueError):SonicJointResidualActionConfig(**kw)


def test_pull_is_bounded_uses_explicit_mappings_and_updates_actual_target_history():
    with wp.ScopedDevice('cpu'):
        target=wp.array([0.,.2,-.2,.6],dtype=wp.float32)
        processed=wp.array([.6,-.2,0.,.2],dtype=wp.float32)
        history=wp.zeros(12,dtype=wp.float32)
        wp.launch(pull_reference_body_targets,dim=(1,2),inputs=[
            wp.array([0],dtype=wp.int32),wp.array([[0.,1.,-1.,0.]],dtype=wp.float32),
            wp.array([0.,0.,.1,0.,0.],dtype=wp.float32),
            wp.array([2,4],dtype=wp.int32),wp.array([1,2],dtype=wp.int32),wp.array([3,1],dtype=wp.int32),
            wp.array([1,1],dtype=wp.int32),1,5,4,4,.5,.15],outputs=[target,processed,history])
        assert np.allclose(target.numpy(),[0.,.35,-.35,.6])
        assert np.allclose(processed.numpy(),[.6,-.35,0.,.35])
        assert np.allclose(history.numpy()[-4:],[0.,-.35,0.,.35])
        assert np.all(history.numpy()[:8]==0)


def test_subset_leaves_other_targets_and_history_exactly_unchanged():
    with wp.ScopedDevice('cpu'):
        target=wp.array([.2,-.2],dtype=wp.float32)
        processed=wp.array([.2,-.2],dtype=wp.float32)
        history=wp.array([.1,.2,.3,.4,.2,-.2],dtype=wp.float32)
        old=history.numpy().copy()
        wp.launch(pull_reference_body_targets,dim=(1,2),inputs=[
            wp.array([0],dtype=wp.int32),wp.array([[1.,-1.]],dtype=wp.float32),
            wp.zeros(2,dtype=wp.float32),wp.array([0,1],dtype=wp.int32),
            wp.array([0,1],dtype=wp.int32),wp.array([0,1],dtype=wp.int32),
            wp.array([1,0],dtype=wp.int32),1,2,2,2,.25,.05],outputs=[target,processed,history])
        assert target.numpy()[0]==pytest.approx(.25)
        assert target.numpy()[1]==np.float32(-.2)
        assert processed.numpy()[1]==np.float32(-.2)
        assert np.array_equal(history.numpy()[:5],old[:5]) is False
        assert np.array_equal(history.numpy()[:4],old[:4])
        assert history.numpy()[5]==old[5]


def test_inference_override_preserves_original_and_rejects_unrelated_edits():
    from real2sim2real.controller_actor_training.controller_override import apply_reference_pull
    config={'action':{'policy_dir':'unchanged'},'env':{'action':{'policy_dir':'unchanged'}},'task':{'parquet':'same'}}
    override=dict(reference_pull_gain=.25,reference_pull_max_rad=.05,
                  reference_pull_joint_names=['right_wrist_yaw_joint'])
    changed=apply_reference_pull(config,override)
    assert changed['action']==changed['env']['action']
    assert changed['task']==config['task']
    assert config['action']=={'policy_dir':'unchanged'}
    for invalid in [dict(override,reference_pull_joint_names=[]),dict(override,reference_pull_joint_names=['bad']),
                    dict(override,reference_pull_joint_names=None),dict(override,reference_pull_gain=2),
                    dict(override,other_setting=1)]:
        with pytest.raises(ValueError):apply_reference_pull(config,invalid)
