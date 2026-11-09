"""One normal tracker update or independent evaluation of external recoveries.

Explicit file handoffs let large proposal models and the tracker use the GPU
sequentially. Neither a failed tracker pose nor an unverified sampler winner is
allowed to update the accepted state.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from .conditioned_recovery import select
from .recovery_quality import rigid,consistency,sample_vertices


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(request,*,cpu_check=False):
    import torch
    from .tracking_recovery import load_exporter,install_rgb_mode
    from real2sim2real.reconstruction_repair.anchored_recovery import visible_occluder
    s=json.loads(Path(request).read_text());root=Path(s['native_inputs']);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    previous=np.asarray(s['previous_pose_camera']);frame=s['source_frame'];before=s['previous_frame']
    mapping=json.loads((root/'source_frame_mapping.json').read_text());ids=mapping['source_frame_indices']
    export=root/'export/source';info=json.loads((export/'wild_export.json').read_text());K=np.asarray(info['intrinsics'])
    if (not rigid(previous) or frame not in ids or not 1<=frame-before<=3
        or s['source_video_sha256']!=info['source_video_sha256']
        or s['mesh_sha256']!=digest(export/'object_mesh/output_aligned.glb')
        or s['fps']!=mapping['source_fps'] or not isinstance(s['depths'],dict)
        or 'original' not in s['depths']
        or any(not isinstance(name,str) or not name for name in s['depths'])):
        raise ValueError('Explicit accepted state, fixed source/mesh and depth comparison required')
    up=load_exporter()
    from prep.mhr_export_utils import frame_names
    names=frame_names(export,0);local=ids.index(frame)
    _,rgb,native_depth,human,obj,k=up._read_foundationpose_frame(export,root/'depth.h5',0,names[local],K)
    occ=json.loads(Path(s['occlusion']).read_text());active=[frame] if frame in occ['active_source_frame_ids'] else []
    occluder=visible_occluder(obj[None],human[None],np.array([frame]),occ['polygon_xy'],active)[0]
    depths={}
    for name,path in s['depths'].items():
        z=np.load(path);d=z['depth_m']
        if (d.shape!=native_depth.shape or d.dtype!=np.float32 or not np.isfinite(d).all() or np.any(d<0)
            or not np.allclose(z['K'],K) or int(z['source_frame'])!=frame):raise ValueError('Native depth grid/frame/camera mismatch')
        depths[name]=d
    if not np.allclose(depths['original'],native_depth,atol=.00001):raise ValueError('Original depth differs from actual native reader')
    support=json.loads((root/'support.json').read_text())
    if support['ground_truth_used'] is not False:raise ValueError('Prediction-only support required')
    mesh=up.load_export_object_mesh(export)
    translation=s.get('depth_guided_translation')
    if translation is not None:
        from real2sim2real.reconstruction_repair.depth_translation import bounded_proposals
        if (not isinstance(translation,dict) or set(translation)!={'max_move_m','target_residual_m'}
            or list(depths)!=['original'] or s['action'] not in ('track','select')):
            raise ValueError('Explicit bounded translation with unchanged original depth required')
        bounded_proposals(mesh.vertices,previous,0.,**translation)
    if cpu_check:
        from omegaconf import OmegaConf
        from .tracking_runtime import MODEL_RUNS
        assert not torch.cuda.is_available();weights=Path(os.environ['FOUNDATIONPOSE_WEIGHTS_DIR'])
        for model in MODEL_RUNS:
            OmegaConf.load(weights/model/'config.yml');state=torch.load(weights/model/'model_best.pth',map_location='cpu',weights_only=True)
            tensors=[v for v in state.get('model',state).values() if torch.is_tensor(v)]
            assert tensors and all(torch.isfinite(v).all() for v in tensors)
        result=dict(passed=True,frame=frame,mesh_faces=len(mesh.faces),native_read=True,models=2,gpu_used=False)
    else:
        install_rgb_mode(up);est=up._pose_estimator(mesh,out/'debug')
        inverse=np.linalg.inv(up._tensor_to_numpy(est.get_tf_to_centered_mesh()));points=sample_vertices(est.mesh_ori.vertices)
        def measure(pose):
            masks=up._render_candidate_masks(est,(np.asarray(pose,np.float32)@inverse)[None],K,obj.shape)
            metrics=up._selected_metric_values(up.measure_visible_silhouettes(masks,occluder,obj),0)
            return {name:dict(metrics=metrics,geometry=consistency(points,pose,K,z,obj&~occluder,support)) for name,z in depths.items()}
        if s['action']=='track':
            est.pose_last=torch.as_tensor((previous@inverse)[None],device='cuda',dtype=torch.float32)
            pose=est.track_one(rgb,depths['original'],K,iteration=5)
            candidates=[dict(name='normal_tracker',pose_camera=pose,provenance=dict(previous_frame=before,source_frame=frame))]
        elif s['action']=='select':
            candidates=[]
            for path in s['proposals']:
                path=Path(path);meta=json.loads(path.with_suffix('.json').read_text());z=np.load(path)
                if (meta['source_frame']!=frame or meta['previous_frame']!=before
                    or not np.array_equal(np.asarray(meta['conditioning_pose_camera']),previous)
                    or meta['source_video_sha256']!=s['source_video_sha256'] or meta['mesh_sha256']!=s['mesh_sha256']
                    or meta['poses_sha256']!=digest(path)):raise ValueError('Proposal is not bound to the accepted tracker state')
                if len(z['poses_camera'])!=len(meta['names']):raise ValueError('Missing candidate identities')
                for name,p in zip(meta['names'],z['poses_camera']):
                    candidates.append(dict(name=meta['depth_hypothesis']+':'+name,pose_camera=p,
                        provenance=dict(source=str(path),sha256=meta['poses_sha256'],previous_frame=before,source_frame=frame)))
        else:raise ValueError('Unknown recovery action')
        result=select(candidates,previous,source_frame=frame,previous_frame=before,fps=s['fps'],measure=measure,
            required_hypotheses=list(depths),recovery=s['action']=='select')
        correction_trace=[]
        if translation is not None and s['action']=='track':
            from .depth_guided_update import correct
            result=correct(pose,previous,vertices=mesh.vertices,source_frame=frame,previous_frame=before,
                fps=s['fps'],measure=measure,**translation)
            correction_trace.append(result)
        if s.get('lookahead'):
            from .recovery_lookahead import qualify
            config=s['lookahead'];future=config['frames']
            if s['action']!='select' or list(depths)!=['original'] or any(f not in ids for f in future):
                raise ValueError('Look-ahead requires original-depth selection and actual future source observations')
            def step(f,pose):
                _,im,z,hm,ob,kf=up._read_foundationpose_frame(export,root/'depth.h5',0,names[ids.index(f)],K)
                active=[f] if f in occ['active_source_frame_ids'] else []
                oc=visible_occluder(ob[None],hm[None],np.array([f]),occ['polygon_xy'],active)[0]
                if not ob.any() or not np.allclose(kf,K):
                    raise ValueError('Look-ahead needs visible masks and the same fixed camera')
                est.pose_last=torch.as_tensor((pose@inverse)[None],device='cuda',dtype=torch.float32)
                p=est.track_one(im,z,K,iteration=5)
                rendered=up._render_candidate_masks(est,(np.asarray(p,np.float32)@inverse)[None],K,ob.shape)
                metric=up._selected_metric_values(up.measure_visible_silhouettes(rendered,oc,ob),0)
                ev=dict(original=dict(metrics=metric,geometry=consistency(points,p,K,z,ob&~oc,support)))
                if translation is not None:
                    from .depth_guided_update import correct
                    def future_measure(q):
                        if np.array_equal(q,p):return ev
                        masks=up._render_candidate_masks(est,(np.asarray(q,np.float32)@inverse)[None],K,ob.shape)
                        measured=up._selected_metric_values(up.measure_visible_silhouettes(masks,oc,ob),0)
                        return dict(original=dict(metrics=measured,geometry=consistency(points,q,K,z,ob&~oc,support)))
                    receipt=correct(p,pose,vertices=mesh.vertices,source_frame=f,previous_frame=f-1,
                        fps=s['fps'],measure=future_measure,**translation)
                    correction_trace.append(receipt)
                    if receipt['accepted']:
                        return receipt['pose_camera'],receipt['candidates'][receipt['selected_index']]['evidence']
                return p,ev
            result=qualify(result,future,fps=s['fps'],step=step,
                max_candidates=config['max_candidates'],horizon=config['horizon'])
        if translation is not None:
            # Copy receipts before attaching them to avoid a self-reference for
            # the current-frame correction result.
            import copy
            result['depth_guided_updates']=copy.deepcopy(correction_trace)
            result['lookahead_tracker_variant']='normal_plus_bounded_depth_translation'
        result.update(action=s['action'],request_sha256=digest(request),native_gpu_measurement=True)
    target=out/('CPU_CHECK.json' if cpu_check else 'RESULTS.json')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(frame=frame,action=s['action'],accepted=result.get('accepted'),cpu_check=cpu_check)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--cpu-check',action='store_true');a=p.parse_args();run(a.request,cpu_check=a.cpu_check)
