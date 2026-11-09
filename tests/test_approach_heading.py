import math
from types import SimpleNamespace

import numpy as np
import pytest
wp = pytest.importorskip("warp", reason="Requires the optional simulation runtime")
pytest.importorskip("newton", reason="Requires the optional simulation runtime")

from flash_chord.runtime.actions.approach_heading import (
    ApproachHeadingConfig, ApproachHeadingController, condition_heading, clear_steering, anchor_legs,
)
from flash_chord.runtime.actions.sonic import SonicJointResidualActionConfig


def condition(yaw=0., desired=0., seconds=2., height=.8, tilt=0., previous=0., gain=.35, encoder=None):
    from scipy.spatial.transform import Rotation
    encoder=np.tile(np.array([1.,0.,0.,1.,0.,0.]),10) if encoder is None else encoder
    values=np.arange(1762,dtype=np.float32)[None]/1000
    values[:,601:661]=encoder
    original=values.copy()
    q=Rotation.from_euler('xyz',[tilt,0,yaw]).as_quat()
    r=Rotation.from_euler('z',desired).as_quat()
    with wp.ScopedDevice('cpu'):
        out=wp.array(values,dtype=wp.float32)
        state=wp.array([previous],dtype=wp.float32)
        tel=wp.zeros((1,4),dtype=wp.float32)
        wp.launch(condition_heading,dim=1,inputs=[wp.array([int(seconds*50)],dtype=wp.int32),
            wp.array([wp.transform((0.,0.,height),q)],dtype=wp.transform),
            wp.array([[0.,0.,.8,*r]],dtype=wp.float32),1,1,0,50.,gain,math.radians(10),
            math.radians(15),.5,10.,12.,.45,math.cos(math.radians(50))],outputs=[state,tel,out])
        return original,out.numpy(),tel.numpy()[0],state.numpy()


def test_default_network_path_is_opt_in_and_invalid_bounds_fail():
    assert SonicJointResidualActionConfig().approach_heading is None
    for kwargs in [dict(heading_gain=1.1),dict(max_heading_rad=.4),dict(heading_gain=True),
                   dict(heading_rate_rad_s=0),dict(leg_pull_gain=.31),dict(end_seconds=9),
                   dict(ramp_seconds=0),dict(min_pelvis_height_m=float('nan'))]:
        with pytest.raises(ValueError):ApproachHeadingConfig(**kwargs)
    with pytest.raises(ValueError):
        SonicJointResidualActionConfig(approach_heading=ApproachHeadingConfig(),reference_pull_gain=.1)


def test_zero_error_preserves_every_token_exactly():
    a,b,tel,_=condition()
    assert np.array_equal(a,b)
    assert tel[1]==0 and tel[2]==0


def test_heading_sign_wrapping_rate_and_token_isolation():
    a,b,tel,_=condition(yaw=math.radians(179),desired=math.radians(-179))
    assert tel[1]==pytest.approx(math.radians(2),abs=1e-6)
    assert tel[2]==pytest.approx(math.radians(15)/50,abs=1e-6)
    assert np.array_equal(a[:,:607],b[:,:607]) # current-orientation identity retained
    assert np.array_equal(a[:,661:],b[:,661:])
    assert b[0,657]>0 # future last rotation R_yx, positive requested yaw
    r=b[0,655:661].reshape(3,2)
    assert np.allclose(r.T@r,np.eye(2),atol=1e-6)


@pytest.mark.parametrize('seconds,height,tilt',[(0,.8,0),(12,.8,0),(30,.8,0),(2,.3,0),(2,.8,1.2)])
def test_start_end_and_fall_suppress_guidance(seconds,height,tilt):
    a,b,tel,_=condition(yaw=-1,seconds=seconds,height=height,tilt=tilt,previous=.15)
    assert np.array_equal(a,b)
    assert tel[0]==0 and tel[2]==0


def test_heading_cap_fade_and_reset_world_isolation():
    _,_,tel,_=condition(yaw=-1,seconds=11,previous=math.radians(10))
    assert tel[2]==pytest.approx(math.radians(5),abs=1e-6)
    state=wp.array([.1,.2],dtype=wp.float32,device='cpu')
    wp.launch(clear_steering,dim=2,inputs=[wp.array([1,0],dtype=wp.int32,device='cpu')],outputs=[state],device='cpu')
    assert np.allclose(state.numpy(),[0,.2])


def test_leg_bound_mappings_subset_and_target_history():
    with wp.ScopedDevice('cpu'):
        target=wp.array([.2,-.2],dtype=wp.float32)
        processed=wp.array([-.2,.2],dtype=wp.float32)
        history=wp.array([1.,2.,3.,4.,-.2,.2],dtype=wp.float32)
        deltas=wp.zeros((1,2),dtype=wp.float32)
        wp.launch(anchor_legs,dim=(1,2),inputs=[wp.array([0],dtype=wp.int32),
            wp.array([[1.,-1.]],dtype=wp.float32),wp.zeros(2,dtype=wp.float32),
            wp.array([0,1],dtype=wp.int32),wp.array([0,1],dtype=wp.int32),
            wp.array([1,0],dtype=wp.int32),wp.array([1,0],dtype=wp.int32),
            wp.array([[.5,0.,0.,0.]],dtype=wp.float32),1,2,2,2,.15,.04],
            outputs=[target,processed,history,deltas])
        assert np.allclose(target.numpy(),[.22,-.2])
        assert np.allclose(processed.numpy(),[-.2,.22])
        assert np.allclose(history.numpy(),[1.,2.,3.,4.,-.2,.22])
        assert np.allclose(deltas.numpy(),[[.02,0]])


def test_real_wrapper_uses_root_body_id_and_only_twelve_leg_joints():
    from flash_chord.runtime.actions.sonic import SONIC_POLICY_JOINT_NAMES
    with wp.ScopedDevice('cpu'):
        action=SimpleNamespace(world_count=1,root_body_id=0,bodies_per_world=1,reference_num_frames=1,
                               encoder_input=wp.zeros((1,1762),dtype=wp.float32))
        scene=SimpleNamespace(robot_reference=SimpleNamespace(fps=50.,joint_q=np.array([[0,0,.8,0,0,0,1]])))
        controller=ApproachHeadingController(scene,action,ApproachHeadingConfig(),'cpu')
        controller.condition(action,wp.array([50],dtype=wp.int32),
                             SimpleNamespace(body_q=wp.array([wp.transform((0,0,.8),(0,0,0,1))],dtype=wp.transform)))
        enabled=controller.enabled.numpy()
        assert enabled.sum()==12
        assert all(not enabled[i] for i,n in enumerate(SONIC_POLICY_JOINT_NAMES)
                   if any(x in n for x in ['waist','wrist','shoulder','elbow']))


def test_diagnostic_wraps_heading_and_exposes_early_drift():
    from scipy.spatial.transform import Rotation
    from real2sim2real.controller_actor_training.body_guidance import summarize
    ref=np.zeros((100,3,7));ref[:,:,2]=.8;ref[:,:,3:]=Rotation.from_euler('z',179,degrees=True).as_quat()
    actual=ref.copy();actual[:,:,0]=np.arange(100)[:,None]/100
    actual[:,:,3:]=Rotation.from_euler('z',-179,degrees=True).as_quat()
    result=summarize(actual,ref,fps=50.,palms=(1,2))
    assert result['heading_rmse_deg']==pytest.approx(2.)
    assert result['first_half_metre_drift_s']==pytest.approx(1.02)
    assert result['first_low_pelvis_s'] is None
    with pytest.raises(ValueError):summarize(actual,ref,fps=0,palms=(1,2))
