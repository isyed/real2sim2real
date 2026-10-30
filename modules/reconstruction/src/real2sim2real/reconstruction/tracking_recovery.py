"""Bounded RGB proposal tracking with independent image/depth/support checks.

GPU dependencies are imported only when installing the process-local adapter.
Upstream exporter and previous artifacts remain unchanged. Failed proposals are
never carried into the next accepted state or a published trajectory.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from .recovery_quality import Limits,consistency,local_proposals,protocol,rejection_reasons,rigid,sample_vertices

def install_recovery(upstream, support, output):
    import torch
    from scipy.spatial.transform import Rotation
    limits = Limits()
    context = {}
    events = []
    original_iterator = upstream._foundationpose_frame_iterator
    original_class = upstream.FoundationPose
    original_dump = upstream._atomic_pickle_dump
    diagnostic = Path(output).with_suffix('.recovery.jsonl')

    def frame_iterator(*args, **kwargs):
        for value in original_iterator(*args, **kwargs):
            frame, rgb, depth, human, obj, K = value
            context.update(frame=int(frame), human=human, object=obj, depth=depth, K=K)
            yield value

    class RecoveringPose(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recovery_points = sample_vertices(self.mesh_ori.vertices)
            self.global_calls = 0
            self.recovery_frames = 0

        def assess(self, poses, previous, *, recovery):
            poses = np.asarray(poses, np.float32).reshape(-1, 4, 4)
            transform = upstream._tensor_to_numpy(self.get_tf_to_centered_mesh())
            centered = poses @ np.linalg.inv(transform)
            masks = upstream._render_candidate_masks(self, centered, context['K'], context['object'].shape)
            measured = upstream.measure_visible_silhouettes(masks, context['human'], context['object'])
            reports = []
            for index, pose in enumerate(poses):
                metrics = upstream._selected_metric_values(measured, index)
                geometry = consistency(self.recovery_points, pose, context['K'], context['depth'],
                    context['object'] & ~context['human'], support)
                reasons = rejection_reasons(pose, previous, metrics, geometry, recovery=recovery, limits=limits)
                reports.append(dict(metrics=metrics, geometry=geometry, rejected_for=reasons))
            return reports

        def choose(self, poses, previous):
            reports = self.assess(poses, previous, recovery=True)
            eligible = [i for i, report in enumerate(reports) if not report['rejected_for']]
            if not eligible:
                return None, None, reports
            chosen = max(eligible, key=lambda i: reports[i]['metrics']['iou'])
            return np.asarray(poses[chosen]).copy(), reports[chosen], reports

        @torch.no_grad()
        def track_one(self, rgb, depth, K, iteration, extra=None):
            transform = upstream._tensor_to_numpy(self.get_tf_to_centered_mesh())
            previous = upstream._tensor_to_numpy(self.pose_last).reshape(4, 4) @ transform
            current = np.asarray(super().track_one(rgb, depth, K, iteration, extra={} if extra is None else extra))
            frame = context['frame']
            if frame != len(events)+1:
                raise ValueError('Recovery context does not match the sequential source frame')
            original = self.assess(current[None], previous, recovery=False)[0]
            event = dict(frame=frame, original=original, recovery=False,
                         method='tracked_update', global_calls_before=self.global_calls)
            selected, report = current, original
            if original['rejected_for']:
                self.recovery_frames += 1
                if self.recovery_frames > limits.max_recovery_frames:
                    raise RuntimeError('Recovery-frame budget exhausted')
                event['recovery'] = True
                proposals = local_proposals(previous, current, K, original['geometry'])
                selected, report, local_reports = self.choose(proposals, previous)
                event['local_candidates'] = len(proposals)
                event['local_best_iou'] = max(p['metrics']['iou'] for p in local_reports)
                event['method'] = 'local_rgb_depth_recovery'
                for rgb_only in (False, True):
                    if selected is not None:
                        break
                    if self.global_calls >= limits.max_global_registrations:
                        break
                    self.pose_last = torch.as_tensor(previous @ np.linalg.inv(transform), device='cuda', dtype=torch.float32)
                    self.register(K=K, rgb=rgb, depth=depth,
                        ob_mask=context['object'] & ~context['human'], iteration=5,
                        rgb_only=rgb_only, seed=0, both_depth_and_rgb=False)
                    self.global_calls += 1
                    poses = upstream._tensor_to_numpy(self.poses) @ transform
                    # Preserve scorer order; evaluate at most 32 candidates that
                    # obey the same step bounds as normal tracking.
                    eligible = []
                    for pose in poses:
                        if not rigid(pose):
                            continue
                        distance = np.linalg.norm(pose[:3, 3]-previous[:3, 3])
                        angle = np.rad2deg(Rotation.from_matrix(pose[:3, :3]@previous[:3, :3].T).magnitude())
                        if distance <= limits.step_translation_m and angle <= limits.step_rotation_deg:
                            eligible.append(pose)
                    event.setdefault('registrations', []).append(dict(rgb_only=rgb_only,
                        total_candidates=len(poses), temporally_eligible=len(eligible)))
                    if eligible:
                        selected, report, _ = self.choose(np.asarray(eligible[:32]), previous)
                        event['method'] = 'rgb_registration' if rgb_only else 'depth_registration'
                if selected is None:
                    event.update(failed=True, global_calls_after=self.global_calls)
                    with diagnostic.open('a') as stream:
                        stream.write(json.dumps(event)+'\n')
                    raise RuntimeError(f'No quality-valid recovery at source frame {frame}; reasons={original["rejected_for"]}')
            event.update(selected=report, global_calls_after=self.global_calls)
            events.append(event)
            with diagnostic.open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            self.pose_last = torch.as_tensor(selected @ np.linalg.inv(transform), device='cuda', dtype=torch.float32)
            return (self.pose_last @ self.get_tf_to_centered_mesh()).cpu().numpy().reshape(4, 4)

    def dump(path, value):
        if 'fp_poses_cam' in value:
            if len(events) != len(value['frames'])-1:
                raise ValueError('Recovery evidence does not cover every tracked source frame')
            value['fp_recovery_applied'] = np.array([False]+[e['recovery'] for e in events], dtype=bool)
            value['metadata'].update(
                source='foundationpose_rgb_seed_with_bounded_tracking_recovery',
                selection_pipeline='quality_checked_tracking_then_bounded_pose_recovery',
                threshold_relaxation=False, unrestricted_candidate_acceptance=False,
                recovery_protocol=protocol(),
                recovery_frame_count=sum(e['recovery'] for e in events),
                recovery_registration_count=events[-1]['global_calls_after'] if events else 0,
                recovery_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                recovery_quality_sha256=hashlib.sha256(Path(__file__).with_name('recovery_quality.py').read_bytes()).hexdigest())
            upstream.validate_foundationpose_selection_metadata(value['metadata'], 'recovery output')
            upstream.validate_foundationpose_selection_arrays(value, (len(value['frames']),), 'recovery output')
        original_dump(path, value)

    upstream.FoundationPose = RecoveringPose
    upstream._foundationpose_frame_iterator = frame_iterator
    upstream._atomic_pickle_dump = dump


def install_rgb_mode(upstream):
    """Change only the proposal refiner; the recovery adapter checks its output."""
    import torch
    original = upstream.FoundationPose

    class RGBPose(original):
        @torch.no_grad()
        def track_one(self, rgb, depth, K, iteration, extra=None):
            prepared = self.prepare_registration_frame(K, rgb, depth,
                np.ones(depth.shape, dtype=bool), tensor_residency=True)
            poses, _ = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors,
                rgb=prepared['rgb_tensor'], depth=prepared['depth_tensor'], K=K,
                ob_in_cams=self.pose_last.reshape(1, 4, 4).detach().cpu().numpy(),
                normal_map=None, xyz_map=prepared['xyz_map_tensor'],
                mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration,
                get_vis=False, rgb_only=True)
            self.pose_last = poses
            return (poses @ self.get_tf_to_centered_mesh()).detach().cpu().numpy().reshape(4, 4)

    upstream.FoundationPose = RGBPose


def install_state_recording(upstream, output):
    original = upstream.FoundationPose
    state_path = Path(output).with_suffix('.states.jsonl')
    if state_path.exists():
        raise FileExistsError('Preserve previous recovery states')

    class RecordedPose(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorded_frame = 0

        def track_one(self, *args, **kwargs):
            self.recorded_frame += 1
            transform = upstream._tensor_to_numpy(self.get_tf_to_centered_mesh())
            before = upstream._tensor_to_numpy(self.pose_last).reshape(4, 4) @ transform
            record = dict(frame=self.recorded_frame, previous_accepted_pose=before.tolist(), accepted=False)
            try:
                result = super().track_one(*args, **kwargs)
                record.update(accepted=True, selected_pose=np.asarray(result).tolist())
                return result
            except BaseException as error:
                record['error'] = str(error)
                raise
            finally:
                with state_path.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')

    upstream.FoundationPose = RecordedPose
    original_dump = upstream._atomic_pickle_dump

    def dump(path, value):
        if 'fp_poses_cam' in value:
            value['metadata'].update(tracking_refiner_mode='rgb_only',
                depth_and_support_checks_retained=True,
                rgb_recovery_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        original_dump(path, value)

    upstream._atomic_pickle_dump = dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['export','depth','seed','support','output']:
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if (Path(args.export)/'poses.npy').exists() or Path(args.output).exists():
        raise ValueError('Require isolated prediction-only inputs and a fresh output')
    support=json.loads(Path(args.support).read_text())
    if support.get('ground_truth_used') is not False or support.get('coordinate_frame')!='camera' or not support.get('evidence'):
        raise ValueError('Require explicit predicted camera-frame support evidence')
    import prep.run_foundationpose_mhr_export as upstream
    install_rgb_mode(upstream)
    install_recovery(upstream,support,args.output)
    install_state_recording(upstream,args.output)
    try:
        upstream.run_foundationpose_mhr_export(args.export,args.depth,args.output,
            camera_id=0,start=0,iteration=5,max_attempts=1,
            first_usable_frame_gt_rotation_oracle=False,
            sequence_mode=upstream.FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK,
            observed_initial_pose=args.seed,viz_file=None)
    except BaseException as error:
        Path(args.output).with_suffix('.failure.json').write_text(json.dumps(dict(error=str(error),protocol=protocol(),tracking_refiner_mode='rgb_only',accepted_reference=False),indent=2)+'\n')
        raise

if __name__=='__main__':main()
