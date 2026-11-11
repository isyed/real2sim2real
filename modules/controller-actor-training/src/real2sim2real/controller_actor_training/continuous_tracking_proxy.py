"""Local object-tracking proxies against a reconstructed continuous reference.

These are not confidential-reference scores or estimates of reconstruction error.
No finger/body imitation metric, mesh registration or symmetry fitting is implied.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json


def interpolate_poses(times, positions, rotations, query):
    t, p, r, q = map(
        lambda v: np.asarray(v, float), (times, positions, rotations, query)
    )
    if (
        t.ndim != 1
        or len(t) < 2
        or p.shape != (len(t), 3)
        or r.shape != (len(t), 3, 3)
        or q.ndim != 1
        or not len(q)
        or not all(np.isfinite(v).all() for v in (t, p, r, q))
        or not (np.diff(t) > 0).all()
        or q.min() < t[0] - 1e-9
        or q.max() > t[-1] + 1e-9
    ):
        raise ValueError(
            "Finite complete timestamp coverage required; no extrapolation"
        )
    if not np.allclose(
        r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-6
    ) or not np.allclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError("Rigid rotations required")
    q = np.clip(q, t[0], t[-1])
    result = np.repeat(np.eye(4)[None], len(q), axis=0)
    result[:, :3, 3] = np.stack([np.interp(q, t, p[:, i]) for i in range(3)], axis=-1)
    result[:, :3, :3] = Slerp(t, Rotation.from_matrix(r))(q).as_matrix()
    return result


def score(achieved, reference, vertices, center):
    from flash_chord.evaluation.metrics import (
        object_add,
        add_auc,
        mean_per_point_error_cm,
        spider_success,
        maniptrans_success,
    )

    a, b = np.asarray(achieved, float).copy(), np.asarray(reference, float).copy()
    v, c = np.asarray(vertices, float), np.asarray(center, float)
    if (
        a.ndim != 3
        or a.shape[1:] != (4, 4)
        or b.shape != a.shape
        or not len(a)
        or v.ndim != 2
        or v.shape[1] != 3
        or not len(v)
        or c.shape != (3,)
        or not all(np.isfinite(x).all() for x in (a, b, v, c))
    ):
        raise ValueError(
            "Finite matching rigid trajectories, vertices and declared centre required"
        )
    for x in (a, b):
        if (
            not np.allclose(x[:, 3], [0, 0, 0, 1])
            or not np.allclose(
                x[:, :3, :3].transpose(0, 2, 1) @ x[:, :3, :3], np.eye(3), atol=1e-6
            )
            or not np.allclose(np.linalg.det(x[:, :3, :3]), 1, atol=1e-6)
        ):
            raise ValueError("Rigid trajectories required")
        x[:, :3, 3] += np.einsum("tij,j->ti", x[:, :3, :3], c)

    def poses(x):
        return np.c_[x[:, :3, 3], Rotation.from_matrix(x[:, :3, :3]).as_quat()]

    ap = poses(a)[:, None, None, :]
    rp = poses(b)[:, None, :]
    add = object_add(ap, rp, (v - c)[None])
    auc, _ = add_auc(add)
    sp, position, orientation = spider_success(ap, rp)
    mp, _, _ = maniptrans_success(ap, rp, np.array([0]))
    return dict(
        add_auc=float(auc),
        mppe_cm=mean_per_point_error_cm(ap, rp),
        sp_sr=float(sp.mean()),
        mp_sr=float(mp.mean()),
        mean_object_position_error_cm=float(position.mean() * 100),
        mean_object_orientation_error_deg=float(np.rad2deg(orientation.mean())),
        mean_add_cm=float(add.mean() * 100),
        single_object_rpe_cm=0.0,
        world_count=1,
        frames=len(a),
        object_count=1,
    )


def run(config):
    import trimesh
    from .guided_motion import load_reference
    from .continuous_reference import reference_goal
    from flash_chord.evaluation import metrics

    result = pinned_json(config["result"])
    path = Path(config["result"]["path"])
    if (
        not result.get("physics_run")
        or result.get("motion_condition") != "demonstrated"
        or not result.get("demonstrated_motion_executed")
        or digest(path.with_name("states.npz")) != result["states_sha256"]
    ):
        raise ValueError("Complete saved demonstrated physics required")
    protocol = result["continuous_protocol"]
    if (
        not protocol["native_timing_preserved"]
        or not protocol["raw_reference_preserved"]
    ):
        raise ValueError("Original continuous reference clock required")
    audit = pinned_json(
        dict(path=result["config"]["audit"], sha256=result["config"]["audit_sha256"])
    )
    parent_path = Path(audit["config"]["trial"])
    # Audit pins its full source inventory, including the input trial.
    if (
        str(parent_path) not in audit["sources"]
        or digest(parent_path) != audit["sources"][str(parent_path)]
    ):
        raise ValueError("Source trial ancestry changed")
    parent = json.loads(parent_path.read_text())
    base = parent["configuration"]
    ref = load_reference(base["reference_guidance"])
    if (
        ref.frame_ids.tolist() != protocol["frame_ids"]
        or abs(ref.elapsed[-1] - protocol["capture_duration_s"]) > 1e-8
    ):
        raise ValueError("Capture frame horizon changed")
    records = result["records"]
    times = np.array([r["time_s"] for r in records])
    positions = np.array([r["object_position_m"] for r in records])
    rotations = np.array([r["object_rotation"] for r in records])
    query = ref.elapsed
    initial_from_source = False
    if times[0] > query[0]:
        first = parent["records"][0]
        if (
            not result.get("object_state_written_only_at_initialization")
            or abs(first["time_s"] - query[0]) > 1e-9
            or times[0] - query[0] > result["physics_timestep_s"] + 1e-9
            or digest(parent_path.with_name("states.npz")) != parent["states_sha256"]
        ):
            raise ValueError("Missing authoritative initialized object pose")
        # Replay initializes object qpos from this pinned kinematic source. Its
        # first measured observation is one physics step later; do not extrapolate.
        times = np.r_[first["time_s"], times]
        positions = np.vstack([first["object_position_m"], positions])
        rotations = np.concatenate(
            [np.asarray(first["object_rotation"])[None], rotations]
        )
        initial_from_source = True
    achieved = interpolate_poses(times, positions, rotations, query)
    reference = np.array(
        [reference_goal(ref, t, protocol["initial_world_from_object"]) for t in query]
    )
    task_path = Path(base["task"])
    if parent["source_hashes"].get(str(task_path)) != digest(task_path):
        raise ValueError("Source task changed")
    task = json.loads(task_path.read_text())
    mesh_path = task_path.parent / task["object"]["mesh"]
    if digest(mesh_path) != task["object"]["sha256"]:
        raise ValueError("Source object mesh changed")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    rng = np.random.default_rng(0)
    ids = rng.choice(len(mesh.vertices), 500, replace=len(mesh.vertices) < 500)
    vertices = np.asarray(mesh.vertices)[ids]
    values = score(achieved, reference, vertices, mesh.centroid)
    origin = score(achieved, reference, vertices, np.zeros(3))
    # No arbitrary fit or time warp. Initial simulator and reconstructed task
    # anchors should already agree; report any discrepancy instead of hiding it.
    initial_translation = float(
        np.linalg.norm(achieved[0, :3, 3] - reference[0, :3, 3])
    )
    sources = {
        str(p): digest(p)
        for p in [
            path,
            parent_path,
            parent_path.with_name("states.npz"),
            Path(result["config"]["audit"]),
            path.with_name("states.npz"),
            task_path,
            mesh_path,
            Path(base["reference_guidance"]["path"]),
            Path(__file__),
            Path(inspect.getfile(metrics)),
        ]
    }
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "sampled-poses.npz",
        achieved=achieved,
        reference=reference,
        frame_ids=ref.frame_ids,
        timestamps_s=query,
        vertices=vertices,
        mesh_surface_centroid_m=mesh.centroid,
    )
    report = dict(
        schema="continuous-tracking-proxy/1",
        config=config,
        metrics=values,
        native_body_origin_sensitivity=origin,
        body_center="Reconstructed area-weighted surface centroid; confidential registered body frame unavailable",
        mesh_vertex_sample_count=500,
        mesh_vertex_seed=0,
        initial_position_discrepancy_m=initial_translation,
        initialized_frame_from_pinned_source=initial_from_source,
        reference_kind="Own reconstructed motion increments at the frozen simulator resting anchor; reconstruction treated as correct",
        official_score=False,
        ground_truth_used=False,
        mesh_cd_o_cm=None,
        reconstruction_error_included=False,
        independent_success_rate_estimated=False,
        physical_acceptance=bool(result["success"]),
        physical_rejected_for=result["rejected_for"],
        release_pass=result["continuous_metrics"]["release"]["accepted"],
        release_tail_excluded_from_pose_proxy=True,
        sources=sources,
        sampled_poses_sha256=digest(out / "sampled-poses.npz"),
    )
    write_json(out / "scores.json", report)
    print(json.dumps(values), flush=True)
    return report


if __name__ == "__main__":
    import os, sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run(json.loads(args.config.read_text()))
    else:
        from real2sim2real.bounded_process import run as bounded

        bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1"},
        )
