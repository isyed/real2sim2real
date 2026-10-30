"""Fresh RGB registration probe with independent initial-pose validity checks.

A supplied or model-ranked first pose is not trusted merely because later
tracking updates are guarded. No temporal step constraint applies before a
first pose exists; image, depth, camera and support checks still apply.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from .recovery_quality import (Limits, consistency, rejection_reasons,
                               sample_vertices)


def choose_seed(poses, metrics, geometry, *, limits=Limits()):
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(metrics) != len(poses) or len(geometry) != len(poses):
        raise ValueError('Initial candidate poses and evidence must have matching lengths')
    reports = [dict(metrics=m, geometry=g,
                    rejected_for=rejection_reasons(p, p, m, g, recovery=True, limits=limits))
               for p, m, g in zip(poses, metrics, geometry)]
    eligible = [i for i, row in enumerate(reports) if not row['rejected_for']]
    return (max(eligible, key=lambda i: metrics[i]['iou']) if eligible else None), reports


def probe(upstream, export, depth, support, output):
    """Use the packaged exporter/model; stop after measuring frame-zero poses."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous registration probes')
    if (Path(export)/'poses.npy').exists():
        raise ValueError('Prediction-only export required')
    if (support.get('ground_truth_used') is not False or support.get('coordinate_frame') != 'camera'
            or not support.get('evidence')):
        raise ValueError('Explicit predicted support provenance required')
    output.mkdir(parents=True)
    original_class = upstream.FoundationPose
    original_iterator = upstream._foundationpose_frame_iterator
    context = {}

    def frame_iterator(*args, **kwargs):
        for value in original_iterator(*args, **kwargs):
            frame, rgb, observed_depth, human, obj, K = value
            if int(frame) != 0:
                raise ValueError('Initial registration probe may only inspect frame zero')
            context.update(depth=observed_depth, human=human, object=obj, K=K)
            yield value

    class ProbeComplete(Exception):
        pass

    class InitialPose(original_class):
        def register(self, *args, **kwargs):
            kwargs.update(rgb_only=True, both_depth_and_rgb=False)
            super().register(*args, **kwargs)
            transform = upstream._tensor_to_numpy(self.get_tf_to_centered_mesh())
            centered = upstream._tensor_to_numpy(self.poses)[:32]
            poses = centered @ transform
            masks = upstream._render_candidate_masks(self, centered, context['K'], context['object'].shape)
            measured = upstream.measure_visible_silhouettes(masks, context['human'], context['object'])
            points = sample_vertices(self.mesh_ori.vertices)
            metrics = [upstream._selected_metric_values(measured, i) for i in range(len(poses))]
            geometry = [consistency(points, p, context['K'], context['depth'],
                                    context['object'] & ~context['human'], support) for p in poses]
            selected, reports = choose_seed(poses, metrics, geometry)
            report = dict(schema='real2sim2real.initial-registration/1', frame=0,
                          inspected_candidates=len(poses), maximum_candidates=32,
                          selected_index=selected, accepted=selected is not None,
                          candidates=reports, limits=Limits().__dict__,
                          later_frames_inspected=False, thresholds_relaxed=False,
                          full_reference_accepted=False, physics_validated=False,
                          implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            np.savez_compressed(output/'candidates.npz', poses_camera=poses)
            if selected is not None:
                seed = dict(source='observed_rgb_silhouette_depth_fit', frame=0,
                            pose_cam=poses[selected].tolist(), K=context['K'].tolist(),
                            clean_trajectory_used=False, method='fresh_rgb_registration_with_independent_validity_gates',
                            probe_report_sha256=hashlib.sha256((output/'report.json').read_bytes()).hexdigest())
                (output/'seed.json').write_text(json.dumps(seed, indent=2)+'\n')
            raise ProbeComplete()

    upstream.FoundationPose = InitialPose
    upstream._foundationpose_frame_iterator = frame_iterator
    try:
        try:
            upstream.run_foundationpose_mhr_export(export, depth, output/'unused.pkl',
                camera_id=0, start=0, end=1, iteration=5, max_attempts=1,
                first_usable_frame_gt_rotation_oracle=False,
                sequence_mode=upstream.FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK,
                observed_initial_pose=None, rgb_only_score=True, viz_file=None)
        except ProbeComplete:
            return json.loads((output/'report.json').read_text())
        raise RuntimeError('Exporter did not execute first-frame registration')
    finally:
        upstream.FoundationPose = original_class
        upstream._foundationpose_frame_iterator = original_iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'depth', 'support', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    from .tracking_recovery import load_exporter
    report = probe(load_exporter(), args.export, args.depth,
                   json.loads(Path(args.support).read_text()), args.output)
    if not report['accepted']:
        raise SystemExit('No first-frame pose passes the unchanged independent validity gates')


if __name__ == '__main__':
    main()
