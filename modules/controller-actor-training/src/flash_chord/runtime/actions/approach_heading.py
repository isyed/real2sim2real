"""Opt-in, bounded early-motion guidance for an unchanged SONIC network.

This is a controller experiment, not a checkpoint compatibility fix. The normal
tokenizer uses reference-relative rotations, as upstream does. Here a bounded
yaw correction is spread across its future horizon (the current token remains
unchanged). Optional leg-target anchoring tests approach drift separately.
Only joint motors act on the robot; no root/object forces or pose resets.
"""
from dataclasses import dataclass
import math

import numpy as np
import warp as wp


@dataclass(frozen=True)
class ApproachHeadingConfig:
    heading_gain: float = 0.35
    max_heading_rad: float = math.radians(10)
    heading_rate_rad_s: float = math.radians(15)
    leg_pull_gain: float = 0.0
    leg_pull_max_rad: float = 0.04
    ramp_seconds: float = 0.5
    fade_start_seconds: float = 10.0
    end_seconds: float = 12.0
    min_pelvis_height_m: float = 0.45
    min_upright_cos: float = math.cos(math.radians(50))

    def __post_init__(self):
        bounds = dict(heading_gain=(0, 1), max_heading_rad=(0, .35),
            heading_rate_rad_s=(0, 1), leg_pull_gain=(0, .3),
            leg_pull_max_rad=(0, .1), ramp_seconds=(0, 5),
            fade_start_seconds=(0, 120), end_seconds=(0, 180),
            min_pelvis_height_m=(0, 1), min_upright_cos=(0, 1))
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"Invalid bounded guidance value: {key}")
        if not 0 < self.ramp_seconds < self.fade_start_seconds < self.end_seconds:
            raise ValueError("Require 0 < ramp < fade start < end")
        if self.heading_rate_rad_s <= 0:
            raise ValueError("Heading rate must be positive")


@wp.kernel
def condition_heading(
    timestep: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    reference_root: wp.array(dtype=wp.float32, ndim=2),
    frames: int, bodies_per_world: int, pelvis_id: int, fps: float,
    gain: float, max_angle: float, rate: float,
    ramp: float, fade_start: float, end: float, min_z: float, min_up: float,
    steering: wp.array(dtype=wp.float32),
    telemetry: wp.array(dtype=wp.float32, ndim=2),
    encoder: wp.array(dtype=wp.float32, ndim=2),
):
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    t = float(timestep[world]) / fps
    root = body_q[world * bodies_per_world + pelvis_id]
    q = wp.transform_get_rotation(root)
    p = wp.transform_get_translation(root)
    rq = wp.quat(reference_root[frame, 3], reference_root[frame, 4],
                 reference_root[frame, 5], reference_root[frame, 6])
    forward = wp.quat_rotate(q, wp.vec3(1., 0., 0.))
    desired = wp.quat_rotate(rq, wp.vec3(1., 0., 0.))
    up = wp.quat_rotate(q, wp.vec3(0., 0., 1.))
    raw = wp.atan2(desired[1], desired[0]) - wp.atan2(forward[1], forward[0])
    error = wp.atan2(wp.sin(raw), wp.cos(raw))
    envelope = wp.min(wp.clamp(t / ramp, 0., 1.),
                      wp.clamp((end - t) / (end - fade_start), 0., 1.))
    if p[2] < min_z or up[2] < min_up:
        envelope = 0.
    goal = wp.clamp(gain * error, -max_angle, max_angle)
    steer = wp.clamp(goal, steering[world] - rate / fps, steering[world] + rate / fps)
    # The envelope bounds the applied correction. It is zero outside the early
    # window and after a fall, regardless of the rate-limited internal value.
    if envelope == 0.:
        steer = 0.
    steering[world] = steer
    applied = steer * envelope
    dx = p[0] - reference_root[frame, 0]
    dy = p[1] - reference_root[frame, 1]
    telemetry[world, 0] = envelope
    telemetry[world, 1] = error
    telemetry[world, 2] = applied
    telemetry[world, 3] = wp.sqrt(dx * dx + dy * dy)
    for future in range(1, 10):
        angle = applied * float(future) / 9.
        c = wp.cos(angle)
        s = wp.sin(angle)
        base = 601 + future * 6
        # Left-compose a world-up yaw in the same relative-rotation convention
        # as upstream. Rotation matrices use row-major first-two-column order.
        for col in range(2):
            x = encoder[world, base + col]
            y = encoder[world, base + 2 + col]
            encoder[world, base + col] = c * x - s * y
            encoder[world, base + 2 + col] = s * x + c * y


@wp.kernel
def anchor_legs(
    timestep: wp.array(dtype=wp.int32), reference: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.array(dtype=wp.float32), q_ids: wp.array(dtype=wp.int32),
    dof_ids: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32), telemetry: wp.array(dtype=wp.float32, ndim=2),
    frames: int, q_per_world: int, dof_per_world: int, processed_dim: int,
    gain: float, bound: float,
    target: wp.array(dtype=wp.float32), processed: wp.array(dtype=wp.float32),
    history: wp.array(dtype=wp.float32), deltas: wp.array(dtype=wp.float32, ndim=2),
):
    world, joint = wp.tid()
    deltas[world, joint] = 0.
    if enabled[joint] == 0 or telemetry[world, 0] == 0.:
        return
    frame = wp.clamp(timestep[world], 0, frames - 1)
    dof = dof_ids[joint]
    at = world * dof_per_world + dof
    desired = reference[frame, dof] + offset[world * q_per_world + q_ids[joint]]
    delta = telemetry[world, 0] * wp.clamp(gain * (desired - target[at]), -bound, bound)
    target[at] += delta
    processed[world * processed_dim + slots[joint]] = target[at]
    history[world * 3 * processed_dim + 2 * processed_dim + slots[joint]] = target[at]
    deltas[world, joint] = delta


@wp.kernel
def clear_steering(mask: wp.array(dtype=wp.int32), steering: wp.array(dtype=wp.float32)):
    world = wp.tid()
    if mask[world] != 0:
        steering[world] = 0.


class ApproachHeadingController:
    def __init__(self, scene, action, config, device):
        self.config = config
        self.fps = float(scene.robot_reference.fps)
        self.root = wp.array(np.ascontiguousarray(scene.robot_reference.joint_q[:, :7]),
                             dtype=wp.float32, device=device)
        self.steering = wp.zeros(action.world_count, dtype=wp.float32, device=device)
        self.telemetry = wp.zeros((action.world_count, 4), dtype=wp.float32, device=device)
        self.deltas = wp.zeros((action.world_count, 29), dtype=wp.float32, device=device)
        from .sonic import SONIC_POLICY_JOINT_NAMES
        joints = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        allowed = {f"{side}_{joint}_joint" for side in ("left", "right") for joint in joints}
        self.enabled = wp.array([int(n in allowed) for n in SONIC_POLICY_JOINT_NAMES],
                                 dtype=wp.int32, device=device)

    def condition(self, action, timestep, state):
        c = self.config
        wp.launch(condition_heading, dim=action.world_count,
            inputs=[timestep, state.body_q, self.root, action.reference_num_frames,
                action.bodies_per_world, action.root_body_id, self.fps,
                c.heading_gain, c.max_heading_rad, c.heading_rate_rad_s, c.ramp_seconds,
                c.fade_start_seconds, c.end_seconds, c.min_pelvis_height_m, c.min_upright_cos],
            outputs=[self.steering, self.telemetry, action.encoder_input])

    def process(self, action, timestep):
        wp.launch(anchor_legs, dim=(action.world_count, 29),
            inputs=[timestep, action.reference_joint_target, action.reference_joint_q_offset,
                action.sonic_q_ids, action.sonic_dof_ids, action.sonic_scalar_slots, self.enabled,
                self.telemetry, action.reference_num_frames, action.q_per_world,
                action.dof_per_world, action.processed_dim, self.config.leg_pull_gain,
                self.config.leg_pull_max_rad],
            outputs=[action.joint_target, action.processed_target, action.actor_action_history, self.deltas])

    def reset(self, mask):
        wp.launch(clear_steering, dim=self.steering.shape[0], inputs=[mask], outputs=[self.steering])
