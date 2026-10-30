"""Bounded RGB proposal tracking with independent image/depth/support checks.

GPU dependencies are imported only when installing the process-local adapter.
Upstream exporter and previous artifacts remain unchanged. Failed proposals are
never carried into the next accepted state or a published trajectory.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from .recovery_quality import Limits,configured_limits,consistency,local_proposals,protocol,rejection_reasons,rigid,sample_vertices


def load_exporter():
    """Select packaged engine source before an image's working-directory copy."""
    import importlib,inspect,sys
    from .backend import engine_paths
    source,body,fp=engine_paths()
    for path in [fp,body,source]:sys.path.insert(0,str(path))
    upstream=importlib.import_module('prep.run_foundationpose_mhr_export')
    expected=(source/'prep/run_foundationpose_mhr_export.py').resolve()
    if Path(upstream.__file__).resolve()!=expected:
        raise ValueError('Loaded exporter is outside the selected engine source: '+str(upstream.__file__))
    inspect.signature(upstream.run_foundationpose_mhr_export).bind(
        'export','depth','output',camera_id=0,start=0,iteration=5,max_attempts=1,
        first_usable_frame_gt_rotation_oracle=False,sequence_mode='register-first-then-track',
        observed_initial_pose='seed.json',viz_file=None)
    return upstream

def install_recovery(upstream, support, output, *, continuous_recovery=None, max_continuous_calls=4,recovery_frame_budget=None):
    if (isinstance(max_continuous_calls,bool) or not isinstance(max_continuous_calls,int)
            or max_continuous_calls<1 or (continuous_recovery is not None and not callable(continuous_recovery))):
        raise ValueError('Explicit callable and positive integer continuous-recovery budget required')
    import torch
    from scipy.spatial.transform import Rotation
    limits = configured_limits(recovery_frame_budget)
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
            self.continuous_calls = 0

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
                         method='tracked_update', global_calls_before=self.global_calls,
                         previous_pose_camera=previous.tolist(),proposed_pose_camera=current.tolist(),
                         proposed_translation_step_m=float(np.linalg.norm(current[:3,3]-previous[:3,3])))
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
                event['local_reports'] = local_reports
                event['local_poses_camera'] = np.asarray(proposals,dtype=np.float32).tolist()
                event['method'] = 'local_rgb_depth_recovery'
                if selected is None and continuous_recovery is not None and self.continuous_calls<max_continuous_calls:
                    # A continuous proposal gets the SAME float32 GPU raster,
                    # depth/support and temporal tests as every other candidate.
                    # Its own CPU acceptance flag is never trusted here.
                    bounded=[i for i,r in enumerate(local_reports)
                             if not set(r['rejected_for']) & {'nonrigid','translation_step','rotation_step','camera_plane'}]
                    if bounded:
                        seed_index=max(bounded,key=lambda i:local_reports[i]['metrics']['iou'])
                        self.continuous_calls+=1
                        try:
                            fitted=continuous_recovery(self.mesh_ori.vertices,self.mesh_ori.faces,previous,proposals[seed_index],
                                K,context['object'],context['human'],context['depth'],support)
                            candidate=np.asarray(fitted['pose_camera'],float)
                            selected,report,checked=self.choose(candidate[None],previous)
                            event['continuous_cpu_recovery']=dict(call=self.continuous_calls,
                                maximum_calls=max_continuous_calls,seed_index=seed_index,fit=fitted,gpu_revalidation=checked)
                            if selected is not None:event['method']='continuous_cpu_proposal_gpu_verified'
                        except (ValueError,KeyError) as error:
                            event['continuous_cpu_recovery']=dict(call=self.continuous_calls,
                                maximum_calls=max_continuous_calls,error=str(error),accepted=False)
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
                recovery_protocol=protocol(recovery_frame_budget=recovery_frame_budget),
                recovery_frame_count=sum(e['recovery'] for e in events),
                recovery_registration_count=events[-1]['global_calls_after'] if events else 0,
                continuous_cpu_recovery_enabled=continuous_recovery is not None,
                continuous_cpu_recovery_max_calls=max_continuous_calls if continuous_recovery is not None else 0,
                continuous_cpu_recovery_count=sum('continuous_cpu_recovery' in e for e in events),
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
    parser.add_argument('--continuous-cpu-recovery',action='store_true',
        help='Opt in to at most four bounded CPU pose fits; every result is revalidated on GPU before use')
    parser.add_argument('--recovery-frame-budget',type=int,
        help='Finite work allowance for failed tracker updates, at most source frames minus one; quality gates are unchanged')
    args=parser.parse_args()
    if (Path(args.export)/'poses.npy').exists() or Path(args.output).exists():
        raise ValueError('Require isolated prediction-only inputs and a fresh output')
    support=json.loads(Path(args.support).read_text())
    if support.get('ground_truth_used') is not False or support.get('coordinate_frame')!='camera' or not support.get('evidence'):
        raise ValueError('Require explicit predicted camera-frame support evidence')
    upstream=load_exporter()
    if args.recovery_frame_budget is not None:
        from prep.mhr_export_utils import frame_names
        configured_limits(args.recovery_frame_budget,source_frames=len(frame_names(Path(args.export),0)))
    install_rgb_mode(upstream)
    continuous=None
    if args.continuous_cpu_recovery:
        # Optional module dependency is loaded only for this explicitly composed
        # reconstruction + repair recovery mode, never for ordinary tracking.
        from real2sim2real.reconstruction_repair.local_registration import propose
        continuous=propose
    install_recovery(upstream,support,args.output,continuous_recovery=continuous,recovery_frame_budget=args.recovery_frame_budget)
    install_state_recording(upstream,args.output)
    try:
        upstream.run_foundationpose_mhr_export(args.export,args.depth,args.output,
            camera_id=0,start=0,iteration=5,max_attempts=1,
            first_usable_frame_gt_rotation_oracle=False,
            sequence_mode=upstream.FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK,
            observed_initial_pose=args.seed,viz_file=None)
    except BaseException as error:
        Path(args.output).with_suffix('.failure.json').write_text(json.dumps(dict(error=str(error),protocol=protocol(recovery_frame_budget=args.recovery_frame_budget),tracking_refiner_mode='rgb_only',accepted_reference=False),indent=2)+'\n')
        raise

if __name__=='__main__':main()
