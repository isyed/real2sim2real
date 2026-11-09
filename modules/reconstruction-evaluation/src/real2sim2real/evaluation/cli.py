"""Score reconstructed motion without exposing references to model workers."""

from pathlib import Path
import argparse
import numpy as np
import trimesh
from real2sim2real.artifacts import digest, write_json
from .geometry import register, sample_surface
from .io import read_reference
from .trajectory import score


def evaluate(
    run,
    reference_motion,
    reference_mesh,
    object_name,
    output,
    *,
    camera_mode="estimated",
):
    run = Path(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "scores.json").exists():
        raise FileExistsError(
            "Keep prior scores; select a new evaluation output directory"
        )
    candidate = trimesh.load(run / "mesh/metric.glb", force="mesh", process=False)
    truth_mesh = trimesh.load(reference_mesh, force="mesh", process=False)
    registration = register(candidate, truth_mesh)
    predicted = np.load(run / "object_motion.npz")
    camera = np.load(run / "camera_motion.npz")
    if not np.array_equal(
        predicted["frame_ids"], camera["frame_ids"]
    ) or not np.array_equal(predicted["timestamps_s"], camera["timestamps_s"]):
        raise ValueError("Object and camera timelines differ")
    truth = read_reference(reference_motion, object_name)
    lookup = {int(fid): i for i, fid in enumerate(truth["frame_ids"])}
    indices = [lookup[int(fid)] for fid in predicted["frame_ids"]]
    if not np.allclose(
        predicted["timestamps_s"], truth["timestamps_s"][indices], atol=1e-6, rtol=0
    ):
        raise ValueError("Prediction/reference clocks differ")
    visible = truth["visible"][indices]
    if not visible[0]:
        raise ValueError("Initial reference frame must be visible")
    if camera_mode not in ("estimated", "fixed"):
        raise ValueError("Unknown camera mode")
    camera_poses = (
        camera["world_from_camera"]
        if camera_mode == "estimated"
        else np.repeat(np.eye(4)[None], len(visible), axis=0)
    )
    poses = camera_poses @ predicted["camera_from_object"]
    metrics, arrays = score(
        poses[visible],
        truth["poses"][indices][visible],
        registration,
        sample_surface(truth_mesh),
        accepted=predicted["accepted"][visible] & camera["accepted"][visible],
    )
    metrics.update(
        mesh_chamfer_cm=registration["chamfer_m"] * 100,
        mesh_registration_scale=registration["scale"],
        reference_visible_frames=int(visible.sum()),
        source_frames=len(visible),
        object_name=object_name,
        comparison="reconstruction_vs_reference",
        score_implementation="local_single_object_v1",
        official_registration_bitwise_parity=False,
        camera_mode=camera_mode,
    )
    write_json(output / "scores.json", metrics)
    write_json(
        output / "mesh_registration.json",
        {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in registration.items()
        },
    )
    write_json(
        output / "provenance.json",
        dict(
            inputs={
                str(p): digest(p)
                for p in [
                    run / "mesh/metric.glb",
                    run / "object_motion.npz",
                    run / "camera_motion.npz",
                    Path(reference_motion),
                    Path(reference_mesh),
                ]
            },
            implementation={
                p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
            },
        ),
    )
    np.savez_compressed(
        output / "errors.npz",
        **arrays,
        frame_ids=predicted["frame_ids"][visible],
        timestamps_s=predicted["timestamps_s"][visible],
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--reference-motion", required=True)
    p.add_argument("--reference-mesh", required=True)
    p.add_argument("--object-name", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--camera-mode",
        choices=["estimated", "fixed"],
        default="estimated",
        help="Fixed is a diagnostic ablation, not a camera estimator",
    )
    a = p.parse_args()
    print(
        evaluate(
            a.run,
            a.reference_motion,
            a.reference_mesh,
            a.object_name,
            a.output,
            camera_mode=a.camera_mode,
        )
    )


if __name__ == "__main__":
    main()
