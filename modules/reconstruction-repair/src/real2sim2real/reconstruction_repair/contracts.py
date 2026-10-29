"""Public typed contracts. Array conventions are enforced by bundle validation."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "r2s2r_engines.repair-initialization/1"
COORDINATES = {
    "length_unit": "m",
    "world_handedness": "right",
    "camera_axes": "opencv_x_right_y_down_z_forward",
    "object_pose": "local_to_world_column_vectors",
    "camera_transform": "world_to_camera_column_vectors",
}
GEOMETRY_KEYS = {
    "vertices", "faces", "object_rotation", "object_translation", "object_scale",
    "frame_ids", "timestamps_s", "intrinsics", "world_to_camera",
}
OBSERVATION_KEYS = {
    "object_mask", "occluder_mask", "reliable", "locked", "hand_points_world",
    "contact_active", "axis_facing",
    "human_mask", "human_occluder_mask", "human_reliable", "human_locked",
    "human_keypoints_xy", "human_keypoints_confidence", "human_contact_active", "human_fit_frames",
}


@dataclass
class BundleData:
    """Writer boundary for a producer adapting its own reconstruction output."""

    metadata: dict[str, Any]
    geometry: dict[str, np.ndarray]
    observations: dict[str, np.ndarray] | None = None
    human: dict[str, np.ndarray] | None = None


@dataclass
class ReconstructionBundle:
    root: Path
    manifest: dict[str, Any]
    geometry: dict[str, np.ndarray]
    observations: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairConfig:
    """One declared, bounded policy. No episode identifiers or frame windows."""

    topology: bool = True
    scale: bool = True
    semantic_orientation: bool = True
    rotation_spikes: bool = True
    pose_recovery: bool = True
    temporal_translation: bool = True
    temporal_preserve_camera_depth: bool = True
    max_gap_s: float = 0.20
    spike_angle_deg: float = 45.0
    neighbor_angle_deg: float = 12.0
    translation_bound_m: float = 0.02
    pose_xy_bound_m: float = 0.05
    pose_rotation_bound_deg: float = 25.0
    pose_trigger_iou: float = 0.65
    anchor_spacing_s: float = 0.5
    max_fit_anchors: int = 24
    fit_max_evaluations: int = 100
    fit_image_size: int = 160
    fit_crop_to_object: bool = True
    pose_motion_backtracking: bool = True
    pose_backtracking_steps: int = 8
    smoothing_window_s: float = 0.15
    max_scale_ratio: float = 2.0
    min_iou_gain: float = 0.002
    max_mean_iou_drop: float = 0.001
    max_frame_iou_drop: float = 0.02
    motion_regression_ratio: float = 1.05
    max_contact_gap_increase_m: float = 0.005
    require_contact_evidence: bool = True
    mesh_vertex_budget: int = 4096
    mesh_face_budget: int = 4096
    human_repair: bool = False
    human_outliers: bool = True
    human_temporal: bool = True
    human_fit: bool = False
    human_root_bound_m: float = 0.02
    human_angle_bound_deg: float = 8.0
    human_joint_bound_m: float = 0.04
    human_depth_bound_m: float = 0.02
    human_hand_bound_m: float = 0.01
    human_root_spike_m: float = 0.008
    human_angle_spike_deg: float = 5.0
    human_confidence_min: float = 0.6
    human_min_joints: int = 8
    human_min_joint_gain_px: float = 0.5
    human_min_joint_gain_fraction: float = 0.1
    human_max_joint_regression_px: float = 2.0
    human_parity_tolerance_m: float = 0.00001
    human_fit_updates: int = 40
    human_fit_learning_rate: float = 0.02
    human_fit_checkpoint_every: int = 10
    human_fit_knot_spacing_s: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RepairConfig":
        defaults = asdict(cls())
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in values.items():
            expected = type(defaults[key])
            if expected is bool and type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            if expected is int and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if expected is float and (type(value) not in (int, float) or not np.isfinite(value)):
                raise ValueError(f"{key} must be a finite number")
        result = cls(**values)
        for spec in fields(result):
            value = getattr(result, spec.name)
            if type(value) is not bool and value <= 0:
                raise ValueError(f"{spec.name} must be positive")
        if result.motion_regression_ratio < 1 or result.max_scale_ratio < 1:
            raise ValueError("Ratio limits must be at least 1")
        if result.pose_trigger_iou > 1 or result.max_frame_iou_drop > 1:
            raise ValueError("IoU limits must not exceed 1")
        if result.human_confidence_min > 1 or result.human_min_joint_gain_fraction > 1:
            raise ValueError("Human confidence and gain fractions must not exceed 1")
        if result.human_min_joints > 12:
            raise ValueError("Only 12 COCO body joints are used for human review")
        if result.human_fit and not result.human_repair:
            raise ValueError("human_fit requires human_repair=true")
        return result


@dataclass
class RepairProposal:
    name: str
    geometry: dict[str, np.ndarray]
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    output_bundle: Path
    report: dict[str, Any]
