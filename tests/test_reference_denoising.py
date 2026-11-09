from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.runtime.actions.reference_denoising import ReferenceDenoisingConfig, condition_reference


def test_preserves_clock_endpoints_other_joints_and_input():
    t = np.arange(700) / 50
    x = np.column_stack([.2*t + .02*np.sin(2*np.pi*12*t), np.sin(t)]).astype(np.float32)
    original = x.copy()
    cfg = ReferenceDenoisingConfig(("leg",))
    y = condition_reference(x, ("leg", "hand"), fps=50, config=cfg)
    assert np.array_equal(x, original)
    assert np.array_equal(y[:, 1], x[:, 1])
    assert np.array_equal(y[0], x[0])
    assert np.array_equal(y[500:], x[500:])
    assert np.max(abs(y-x)) <= cfg.max_change_rad
    assert np.std(np.diff(y[50:350, 0])) < np.std(np.diff(x[50:350, 0]))*.7


def test_excessive_change_is_rejected_without_mutation():
    x = np.zeros((700, 1)); x[100] = 2
    original = x.copy()
    with pytest.raises(ValueError, match="exceeds"):
        condition_reference(x, ("leg",), fps=50, config=ReferenceDenoisingConfig(("leg",)))
    assert np.array_equal(x, original)


@pytest.mark.parametrize("kw", [{"strength": True}, {"sigma_seconds": float("nan")},
    {"end_seconds": 7}, {"max_change_rad": .2}, {"joint_names": ("leg", "leg")}])
def test_invalid_configuration(kw):
    with pytest.raises(ValueError):
        ReferenceDenoisingConfig(**dict({"joint_names": ("leg",)}, **kw))


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_invalid_clock(fps):
    with pytest.raises(ValueError):
        condition_reference(np.zeros((100,1)), ("leg",), fps=fps,
                            config=ReferenceDenoisingConfig(("leg",)))


def test_encoder_changes_only_selected_positions_and_consistent_velocities():
    pytest.importorskip("newton", reason="Requires the optional simulation runtime")
    from flash_chord.runtime.actions.sonic import (
        SONIC_POLICY_JOINT_NAMES, SonicJointResidualActionConfig, build_sonic_encoder_reference)
    names = SONIC_POLICY_JOINT_NAMES
    t = np.arange(700)/50
    q = np.zeros((700, 36), dtype=np.float32);q[:,6]=1
    q[:,7]=.1*t+.02*np.sin(2*np.pi*12*t)
    scene = SimpleNamespace(robot_reference=SimpleNamespace(joint_q=q, fps=50.,num_frames=len(q)),
                            layout=SimpleNamespace(scalar_joints=SimpleNamespace(names=names,q_ids=np.arange(7,36))))
    a=build_sonic_encoder_reference(scene)
    assert SonicJointResidualActionConfig().reference_denoising is None
    assert np.array_equal(a,build_sonic_encoder_reference(scene,None))
    cfg=ReferenceDenoisingConfig((names[0],))
    b=build_sonic_encoder_reference(scene,cfg)
    expected=condition_reference(q[:,7:],names,fps=50,config=cfg)
    frames=np.minimum(np.arange(700)[:,None]+np.arange(10)[None]*5,699)
    vel=np.zeros_like(expected);vel[:-1]=np.diff(expected,axis=0)*50
    np.testing.assert_array_equal(b[:,4:294],expected[frames].reshape(700,-1))
    np.testing.assert_array_equal(b[:,294:584],vel[frames].reshape(700,-1))
    assert np.array_equal(a[:,584:],b[:,584:])
    assert np.array_equal(a[:,0:4],b[:,0:4])
    assert np.array_equal(a[500:],b[500:])
    assert np.array_equal(q[:,7],(.1*t+.02*np.sin(2*np.pi*12*t)).astype(np.float32))
