"""CPU tests for the optional heuristic; these do not establish grasp success."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import warp as wp

from flash_chord.runtime.actions.hand_closing import (
    HandClosingConfig, HandClosingController, joint_endpoints, reference_schedule,
)
from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES


def scene_and_action():
    def array(x, dtype=float):
        return wp.array(np.asarray(x), dtype=dtype, device="cpu")
    names = SONIC_DIRECT_JOINT_NAMES
    # Translate and rotate object 90 degrees: local +x palm maps to world +y.
    reference = NS(num_frames=3, fps=50.0,
        object_body_pos_w=lambda: np.tile([[[1., 0., 0.]]], (3, 1, 1)),
        object_body_quat_w=lambda: np.tile([[[2**-.5, 0., 0., 2**-.5]]], (3, 1, 1)),
        contact_active=lambda side: np.array([1, 1, 0]) if side == 'left' else np.array([0, 1, 0]))
    palms = [NS(side=s, wrist_pos_w=np.tile([1., 1., 0.], (3, 1))) for s in ('left', 'right')]
    scene = NS(source_reference=reference, world_count=2,
        robot_reference=NS(num_frames=3, fps=50., hands=palms),
        objects=[NS(root=NS(body_id=2, reference_body_id=0))], object_scales=np.ones((2, 1)),
        layout=NS(hands=[NS(side=s, palm_frame=NS(body_id=i, body_to_frame_pos=(0., 0., 0.)))
                         for i, s in enumerate(('left', 'right'))]),
        model=NS(joint_limit_lower=array([-1.] * 43), joint_limit_upper=array([1.] * 43)))
    action = NS(finger_joint_names=names, finger_dof_ids=array(range(29, 43), int),
        finger_q_ids=array(range(29, 43), int), finger_scalar_slots=array(range(29, 43), int),
        bodies_per_world=3, q_per_world=43, dof_per_world=43, finger_residual_scale=.15,
        use_tanh=False, action_dim=43, residual_joint_names=tuple(range(29)), processed_dim=43,
        joint_target=array([7.] * 86), processed_target=array([7.] * 86),
        actor_action_history=array([7.] * (2 * 3 * 43)))
    state = NS(body_q=array([[1., 1., 0., 0., 0., 0., 1.],
                            [1., 1., 0., 0., 0., 0., 1.],
                            [1., 0., 0., 0., 0., 2**-.5, 2**-.5]] * 2, wp.transform),
               joint_q=array([0.] * 86))
    return scene, action, state, array


def test_named_joint_endpoints_match_upstream_mirrored_thumb_formula():
    names = SONIC_DIRECT_JOINT_NAMES
    lower = np.arange(14) * .03 - 1
    upper = lower + 2
    opened, closed = joint_endpoints(names, lower, upper)
    for fraction in (0., .25, 1.):
        for i, name in enumerate(names):
            amount = 1 - fraction if 'thumb' in name else fraction
            upstream = (amount * (lower[i] - upper[i]) + upper[i] if name.startswith('left_')
                        else amount * (upper[i] - lower[i]) + lower[i])
            assert opened[i] + fraction * (closed[i] - opened[i]) == pytest.approx(upstream)


def test_object_local_anchors_and_independent_contact_phase():
    scene, _, _, _ = scene_and_action()
    phase, anchor = reference_schedule(scene)
    np.testing.assert_array_equal(phase, [[1, 0], [1, 1], [0, 0]])
    np.testing.assert_allclose(anchor, np.tile([[[1., 0., 0.], [1., 0., 0.]]], (3, 1, 1)), atol=1e-6)
    scene.robot_reference.fps = 30
    with pytest.raises(ValueError, match='timing'):
        reference_schedule(scene)


def test_kernel_release_residual_limits_history_and_partial_reset():
    scene, owner, state, array = scene_and_action()
    cfg = HandClosingConfig(closure_rate=50., target_rate=1000.)
    controller = HandClosingController(scene, owner, cfg, "cpu")
    action = np.zeros((2, 43), dtype=np.float32)
    action[0, 29] = 2.  # learned correction must survive the closing prior
    action[1, 29:] = 100.  # final targets must respect limits
    controller.process(array(action.ravel()), array([0, 1], int), state, owner)
    np.testing.assert_array_equal(controller.closure.numpy(), [[1, 0], [1, 1]])
    targets = owner.joint_target.numpy().reshape(2, 43)
    assert targets[0, 29] == pytest.approx(-.7)  # left index closed=-1 + .15*2
    np.testing.assert_array_equal(targets[:, :29], 7.)  # body controller untouched
    assert np.max(np.abs(targets[:, 29:])) <= 1.
    history = owner.actor_action_history.numpy().reshape(2, 3, 43)
    np.testing.assert_array_equal(history[:, :2], 7.)
    np.testing.assert_array_equal(history[:, -1], owner.processed_target.numpy().reshape(2, 43))
    controller.process(array(np.zeros(86)), array([2, 2], int), state, owner)
    np.testing.assert_array_equal(controller.closure.numpy(), 0.)
    controller.process(array(np.zeros(86)), array([1, 1], int), state, owner)
    controller.reset(array([1, 0], int))
    np.testing.assert_array_equal(controller.closure.numpy(), [[0, 0], [1, 1]])
    np.testing.assert_array_equal(controller.initialized.numpy(), [[0, 0], [1, 1]])


def test_default_rates_bound_first_target_and_closure_changes():
    scene, owner, state, array = scene_and_action()
    controller = HandClosingController(scene, owner, HandClosingConfig(), "cpu")
    controller.process(array(np.zeros(86)), array([1, 1], int), state, owner)
    np.testing.assert_allclose(controller.closure.numpy(), .08)
    assert np.max(np.abs(owner.joint_target.numpy().reshape(2, 43)[:, 29:])) <= .120001


def test_hydra_optional_configuration_defaults_off():
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.runtime.actions.sonic import SonicJointResidualActionConfig
    baseline = compose_config('train_flash_sac', ['experiment=g1_recon_body_flash_sac'])
    assert instantiate_typed(baseline.action, SonicJointResidualActionConfig).hand_closing is None
    enabled = compose_config('train_flash_sac', ['experiment=g1_recon_body_flash_sac',
        '+action.hand_closing._target_=flash_chord.runtime.actions.hand_closing.HandClosingConfig'])
    assert isinstance(instantiate_typed(enabled.action, SonicJointResidualActionConfig).hand_closing, HandClosingConfig)
