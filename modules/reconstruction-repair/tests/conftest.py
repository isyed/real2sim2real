"""Procedural fixtures contain no challenge videos, labels or licensed assets."""

from dataclasses import asdict

import numpy as np
import pytest

from real2sim2real.reconstruction_repair.bundle import bundle_create
from real2sim2real.reconstruction_repair.contracts import BundleData, COORDINATES, RepairConfig
from real2sim2real.reconstruction_repair.geometry import mesh_silhouette


def diagnostic_repair_run(*args, **kwargs):
    """Explicitly isolate fine-algorithm tests from the mandatory metric workflow.

    These tests have no metric depth evidence; their outputs are now labelled
    unverified diagnostics. Default fail-closed behavior has separate tests.
    """
    from real2sim2real.reconstruction_repair.runner import repair_run
    return repair_run(*args, **kwargs, diagnostic_only=True)


def box_geometry(n=9):
    vertices = np.array([[-.16, -.05, -.03], [.16, -.05, -.03], [.16, .05, -.03], [-.16, .05, -.03],
                         [-.16, -.05, .03], [.16, -.05, .03], [.16, .05, .03], [-.16, .05, .03]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return {"vertices": vertices, "faces": faces, "object_scale": np.array(1.0),
            "object_rotation": np.tile(np.eye(3), (n, 1, 1)),
            "object_translation": np.tile([0., 0., 2.], (n, 1)),
            "frame_ids": np.arange(n, dtype=np.int64) + 100, "timestamps_s": np.arange(n) / 30.,
            "intrinsics": np.tile(np.array([[400., 0., 64.], [0., 400., 48.], [0., 0., 1.]]), (n, 1, 1)),
            "world_to_camera": np.tile(np.eye(4), (n, 1, 1))}


def observations(g):
    n = len(g["frame_ids"])
    return {"object_mask": np.stack([mesh_silhouette(g, i, 96, 128) for i in range(n)]),
            "reliable": np.ones(n, bool), "locked": np.zeros(n, bool),
            "hand_points_world": np.tile(np.array([[[0., 0., 2.]]]), (n, 1, 1)),
            "contact_active": np.ones((n, 1), bool)}


def metadata():
    return {"sequence_id": "previously-unseen-sequence", "world_frame_id": "synthetic-camera-rig",
            "coordinates": COORDINATES, "human": {"encoding": "test-only-numeric-payload"},
            "evidence": {"hand_points_source": "procedural regression fixture, not real contact"}}


def only(*stages, **overrides):
    values = asdict(RepairConfig())
    for key in ("topology", "scale", "semantic_orientation", "rotation_spikes", "pose_recovery", "temporal_translation"):
        values[key] = key in stages
    values.update(overrides)
    return RepairConfig.from_dict(values)


@pytest.fixture
def make_bundle(tmp_path):
    count = 0

    def create(g=None, o=None, m=None, human=None):
        nonlocal count
        count += 1
        g = box_geometry() if g is None else g
        return bundle_create(tmp_path / f"input-{count}", BundleData(metadata() if m is None else m, g,
                             observations(g) if o is None else o,
                             {"pose": np.arange(len(g["frame_ids"]) * 5, dtype=float).reshape(-1, 5)} if human is None else human))
    return create
