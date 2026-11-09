"""Portable interaction artifacts distinguish partial diagnostics from complete hands."""

import json
from pathlib import Path
import shutil

import numpy as np

from real2sim2real.artifacts import digest, publish
from .interaction_io import Reconstruction
from .camera import transform


def validate_hands(hands, scene):
    """Check shared time, accepted/missing semantics and world-frame composition."""
    if not np.array_equal(hands["frame_ids"], scene["frame_ids"]) or not np.allclose(
        hands["timestamps_s"], scene["timestamps_s"], atol=1e-8, rtol=0
    ):
        raise ValueError("Hand and scene timelines differ")
    count = len(scene["frame_ids"])
    accepted = hands["accepted"]
    if accepted.shape != (count, 2) or accepted.dtype != bool:
        raise ValueError("Expected per-frame left/right hand validity")
    for key in ("vertices", "joints"):
        world, camera = hands["world_" + key], hands["camera_" + key]
        if (
            world.ndim != 4
            or world.shape[:2] != (count, 2)
            or world.shape[-1] != 3
            or camera.shape != world.shape
        ):
            raise ValueError("Invalid hand geometry shape")
        if (
            not np.isfinite(world[accepted]).all()
            or not np.isfinite(camera[accepted]).all()
        ):
            raise ValueError("Accepted hands must be finite")
        if (
            not np.isnan(world[~accepted]).all()
            or not np.isnan(camera[~accepted]).all()
        ):
            raise ValueError("Missing hands must remain NaN")
        for i, h in np.argwhere(accepted):
            expected = transform(camera[i, h], scene["world_from_camera"][i])
            if not np.allclose(world[i, h], expected, atol=1e-5, rtol=0):
                raise ValueError("Hands and objects do not share the declared world")


def publish_scene(config):
    root = Path(config["output"])
    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    support = json.loads((root / "support/support.json").read_text())
    if not support["accepted"]:
        raise ValueError("A rejected support cannot be published")
    hand_path = root / "hands_metric/hands.npz"
    if not hand_path.exists():
        raise ValueError(
            "Articulated 3D hands are required; use the partial review while model assets are missing"
        )
    hands = np.load(hand_path)
    validate_hands(hands, np.load(root / "support/scene.npz"))
    if not np.array_equal(hands["frame_ids"], source.frame_ids):
        raise ValueError("Hand frames differ")
    if not hands["accepted"].any(0).all():
        raise ValueError("Both hands require measured, accepted estimates")
    diagnostics = json.loads((root / "contacts/diagnostics.json").read_text())
    if diagnostics["summary"]["sources"]["hands"] != digest(hand_path):
        raise ValueError("Contact diagnostics do not describe these hands")
    files = dict(
        scene=root / "support/scene.npz",
        support=root / "support/support.json",
        table_collision=root / "support/tabletop.glb",
        hands=hand_path,
        hand_quality=root / "hands_metric/quality.json",
        diagnostics=root / "contacts/diagnostics.json",
    )
    target = root / "object.glb"
    if target.exists():
        raise FileExistsError("Preserve existing scene mesh")
    shutil.copyfile(source.path("mesh/metric.glb"), target)
    if not (root / "inputs.json").exists():
        source.receipt(
            root,
            mask_ids=[
                config.get("object_id", 1),
                *config["hand_landmarks"].get("mask_ids", [2, 3]),
            ],
            config=config,
        )
    files.update(object_mesh=target, inputs=root / "inputs.json")
    return publish(
        root,
        "metric-interaction-reconstruction",
        files,
        metadata=dict(
            units="metres",
            up_axis="+Z",
            world_frame="fixed_observed_tabletop",
            hand_order=["left", "right"],
            hand_reconstruction=True,
            complete=bool(hands["accepted"].all()),
            accepted_hand_frames=hands["accepted"].sum(0).tolist(),
            missing_hand_frames="NaN with accepted=false; never interpolated",
            gravity_source="level_table_assumption",
            control_ready=False,
        ),
    )
