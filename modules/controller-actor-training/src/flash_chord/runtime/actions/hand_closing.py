# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Dex3 closing prior; learned finger residuals remain active.

Ports G1GraspPolicy's joint-limit interpolation and thumb/side conventions.
Release follows each hand's reference contact phase, not distance to the final
object position. Proximity is measured to the reference palm expressed in the
current object's frame, avoiding an object-centre threshold for large objects.
This is a heuristic prior, not a learned grasp policy or a grasp optimiser.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class HandClosingConfig:
    start_distance: float = 0.20
    closed_distance: float = 0.05
    closure_rate: float = 4.0  # fraction of full closure / second
    target_rate: float = 6.0  # radians / second, including learned residual

    def __post_init__(self):
        values = (self.start_distance, self.closed_distance, self.closure_rate, self.target_rate)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("hand-closing settings must be finite")
        if not 0 <= self.closed_distance < self.start_distance:
            raise ValueError("require 0 <= closed_distance < start_distance")
        if min(self.closure_rate, self.target_rate) <= 0:
            raise ValueError("hand-closing rates must be positive")


def joint_endpoints(names, lower, upper):
    """Match the executable upstream formula, including mirrored thumb signs."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (len(names),) or upper.shape != lower.shape:
        raise ValueError("one pair of limits is required per named finger joint")
    if not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper):
        raise ValueError("finger joint limits must be finite and ordered")
    if any(not n.startswith(("left_hand_", "right_hand_")) for n in names):
        raise ValueError("expected named left/right Dex3 finger joints")
    closes_to_upper = np.array([n.startswith("right_") != ("thumb" in n) for n in names])
    return np.where(closes_to_upper, lower, upper), np.where(closes_to_upper, upper, lower)


def reference_schedule(scene):
    """Use the already resampled task reference; never reload at a different FPS."""
    reference = scene.source_reference
    if reference is None or len(scene.objects) != 1:
        raise ValueError("hand closing currently requires a task reference and one logical object")
    count = scene.robot_reference.num_frames
    if reference.num_frames != count or reference.fps != scene.robot_reference.fps:
        raise ValueError("hand phase and robot reference timing must match")
    obj = scene.objects[0].root
    position = reference.object_body_pos_w()[:, obj.reference_body_id]
    quat = reference.object_body_quat_w()[:, obj.reference_body_id]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.all(np.isfinite(norm)):
        raise ValueError("invalid reference object orientation")
    quat = quat / norm
    hands = {h.side: h for h in scene.robot_reference.hands}
    phases, anchors = [], []
    for side in ("left", "right"):
        phase = np.asarray(reference.contact_active(side), dtype=np.float32)
        if phase.shape != (count,) or not np.all(np.isin(phase, [0, 1])):
            raise ValueError(f"{side} contact phase must be binary and frame aligned")
        # Inverse rotation: reference palm minus reference object position.
        delta = hands[side].wrist_pos_w - position
        xyz, w = -quat[:, 1:], quat[:, :1]
        local = delta + 2 * np.cross(xyz, np.cross(xyz, delta) + w * delta)
        phases.append(phase)
        anchors.append(local)
    phase = np.stack(phases, axis=1).astype(np.float32)
    anchor = np.stack(anchors, axis=1).astype(np.float32)
    if not np.all(np.isfinite(anchor)):
        raise ValueError("reference palm anchors must be finite")
    return phase, anchor


@wp.kernel
def apply_hand_closing(
    body_q: wp.array(dtype=wp.transform), joint_q: wp.array(dtype=float),
    timestep: wp.array(dtype=int), action: wp.array(dtype=float),
    phase: wp.array2d(dtype=float), anchor: wp.array2d(dtype=wp.vec3),
    palm_body: wp.array(dtype=int), palm_local: wp.array(dtype=wp.vec3),
    object_body: int, bodies_per_world: int, q_per_world: int, dof_per_world: int,
    object_scale: wp.array(dtype=float), finger_q: wp.array(dtype=int),
    finger_dof: wp.array(dtype=int), finger_slot: wp.array(dtype=int),
    finger_side: wp.array(dtype=int), lower: wp.array(dtype=float), upper: wp.array(dtype=float),
    opened: wp.array(dtype=float), closed: wp.array(dtype=float),
    start_distance: float, closed_distance: float, closure_step: float, target_step: float,
    residual_scale: float, use_tanh: bool, action_dim: int, body_dim: int,
    processed_dim: int, history_length: int,
    closure: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
    initialized: wp.array2d(dtype=int), joint_target: wp.array(dtype=float),
    processed: wp.array(dtype=float), history: wp.array(dtype=float),
):
    world, side = wp.tid()
    frame = wp.clamp(timestep[world], 0, phase.shape[0] - 1)
    palm = wp.transform_point(body_q[world * bodies_per_world + palm_body[side]], palm_local[side])
    target_palm = wp.transform_point(body_q[world * bodies_per_world + object_body],
                                     anchor[frame, side] * object_scale[world])
    distance = wp.length(palm - target_palm)
    desired = phase[frame, side] * wp.clamp((start_distance - distance) /
                                           (start_distance - closed_distance), 0.0, 1.0)
    amount = closure[world, side] + wp.clamp(desired - closure[world, side], -closure_step, closure_step)
    closure[world, side] = amount
    for finger in range(finger_q.shape[0]):
        if finger_side[finger] == side:
            residual = action[world * action_dim + body_dim + finger]
            if use_tanh:
                residual = wp.tanh(residual)
            target = opened[finger] + amount * (closed[finger] - opened[finger]) + residual_scale * residual
            target = wp.clamp(target, lower[finger], upper[finger])
            old = previous[world, finger]
            if initialized[world, side] == 0:
                old = joint_q[world * q_per_world + finger_q[finger]]
            target = old + wp.clamp(target - old, -target_step, target_step)
            target = wp.clamp(target, lower[finger], upper[finger])
            previous[world, finger] = target
            joint_target[world * dof_per_world + finger_dof[finger]] = target
            processed[world * processed_dim + finger_slot[finger]] = target
            # SONIC already shifted the history; replace its newest finger entry.
            history[(world * history_length + history_length - 1) * processed_dim + finger_slot[finger]] = target
    initialized[world, side] = 1


@wp.kernel
def reset_hand_closing(mask: wp.array(dtype=int), closure: wp.array2d(dtype=float),
                       initialized: wp.array2d(dtype=int)):
    world, side = wp.tid()
    if mask[world] != 0:
        closure[world, side] = 0.0
        initialized[world, side] = 0


class HandClosingController:
    def __init__(self, scene, action, config, device):
        self.config, self.world_count = config, scene.world_count
        phase, anchor = reference_schedule(scene)
        names = action.finger_joint_names
        dofs = action.finger_dof_ids.numpy()
        lower = scene.model.joint_limit_lower.numpy()[dofs]
        upper = scene.model.joint_limit_upper.numpy()[dofs]
        opened, closed = joint_endpoints(names, lower, upper)
        hands = {h.side: h.palm_frame for h in scene.layout.hands}
        def array(value, dtype):
            return wp.array(np.asarray(value), dtype=dtype, device=device)
        self.static_inputs = [
            array(phase, float), array(anchor, wp.vec3),
            array([hands[s].body_id for s in ("left", "right")], int),
            array([hands[s].body_to_frame_pos for s in ("left", "right")], wp.vec3),
            scene.objects[0].root.body_id, action.bodies_per_world, action.q_per_world, action.dof_per_world,
            array(scene.object_scales[:, 0], float), action.finger_q_ids, action.finger_dof_ids,
            action.finger_scalar_slots, array([int(n.startswith("right_")) for n in names], int),
            array(lower, float), array(upper, float), array(opened, float), array(closed, float),
            config.start_distance, config.closed_distance, config.closure_rate / scene.robot_reference.fps,
            config.target_rate / scene.robot_reference.fps, action.finger_residual_scale, action.use_tanh,
            action.action_dim, len(action.residual_joint_names), action.processed_dim, 3,
        ]
        self.closure = wp.zeros((scene.world_count, 2), dtype=float, device=device)
        self.previous = wp.zeros((scene.world_count, len(names)), dtype=float, device=device)
        self.initialized = wp.zeros((scene.world_count, 2), dtype=int, device=device)

    def process(self, action, timestep, state, owner):
        wp.launch(apply_hand_closing, dim=(self.world_count, 2), inputs=[
            state.body_q, state.joint_q, timestep, action, *self.static_inputs,
            self.closure, self.previous, self.initialized, owner.joint_target,
            owner.processed_target, owner.actor_action_history,
        ], device=owner.joint_target.device)

    def reset(self, mask):
        wp.launch(reset_hand_closing, dim=(self.world_count, 2),
                  inputs=[mask, self.closure, self.initialized], device=self.closure.device)
