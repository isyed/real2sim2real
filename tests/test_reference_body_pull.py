import numpy as np
import pytest
import warp as wp
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
            1,5,4,4,.5,.15],outputs=[target,processed,history])
        assert np.allclose(target.numpy(),[0.,.35,-.35,.6])
        assert np.allclose(processed.numpy(),[.6,-.35,0.,.35])
        assert np.allclose(history.numpy()[-4:],[0.,-.35,0.,.35])
        assert np.all(history.numpy()[:8]==0)
