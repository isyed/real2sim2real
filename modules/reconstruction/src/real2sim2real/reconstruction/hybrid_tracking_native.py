"""Native FoundationPose comparison with optional saved external pose proposals.

Dataset-independent, prediction-only inputs. The source camera must be fixed.
Outputs are diagnostic sparse poses, never an automatically accepted artifact.
"""
import argparse,json,os,time,hashlib
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def execute(inputs,proposals,metadata,seed,occlusion,output,*,cpu_check=False):
    import torch
    from omegaconf import OmegaConf
    from .tracking_recovery import load_exporter,install_rgb_mode
    from .tracking_runtime import MODEL_RUNS
    from .recovery_quality import sample_vertices,consistency,rigid
    from .hybrid_tracking import HybridConfig,run_sequence
    from real2sim2real.reconstruction_repair.anchored_recovery import visible_occluder
    up=load_exporter()
    from prep.mhr_export_utils import frame_names
    inputs,proposals,metadata,seed,occlusion,output=map(Path,[inputs,proposals,metadata,seed,occlusion,output])
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    export=inputs/'export/source';info=json.loads((export/'wild_export.json').read_text())
    mapping=json.loads((inputs/'source_frame_mapping.json').read_text());meta=json.loads(metadata.read_text())
    anchor=json.loads(seed.read_text());occ=json.loads(occlusion.read_text());support=json.loads((inputs/'support.json').read_text())
    ids=np.asarray(mapping['source_frame_indices']);K=np.asarray(info['intrinsics']);names=frame_names(export,0)
    if (export/'poses.npy').exists():raise ValueError('Clean trajectories are not permitted')
    if (len(ids)!=len(names) or not np.array_equal(ids,np.arange(ids[0],ids[-1]+1))
        or meta['source_video_sha256']!=info['source_video_sha256']
        or meta['mesh_sha256']!=digest(export/'object_mesh/output_aligned.glb')
        or meta['coordinate_frame']!='camera' or meta['fixed_camera'] is not True
        or anchor['source_frame']!=int(ids[0]) or not rigid(anchor['pose_cam'])
        or not np.allclose(anchor['K'],K) or support['ground_truth_used'] is not False):
        raise ValueError('Source identity, native mapping, fixed camera or anchor contract differs')
    z=np.load(proposals);pids=z['source_frame_ids'];poses=z['poses_camera']
    if (poses.shape!=(len(pids),4,4) or not all(rigid(p) for p in poses)
        or not np.isin(pids,ids).all() or len(meta['rows'])!=len(pids)
        or meta['proposals_sha256']!=digest(proposals)):
        raise ValueError('Invalid or unbound external proposal cache')
    frames={};observed=[]
    for local,native in enumerate(ids):
        _,rgb,depth,human,obj,k=up._read_foundationpose_frame(export,inputs/'depth.h5',0,names[local],K)
        if not np.allclose(k,K):raise ValueError('Changing camera requires explicit transforms')
        active=[int(native)] if int(native) in occ['active_source_frame_ids'] else []
        occluder=visible_occluder(obj[None],human[None],np.array([native]),occ['polygon_xy'],active)[0]
        frames[int(native)]=(rgb,depth,occluder,obj,k);observed.append(bool(obj.any()))
    mesh=up.load_export_object_mesh(export)
    if cpu_check:
        assert not torch.cuda.is_available()
        checked=[];root=Path(os.environ['FOUNDATIONPOSE_WEIGHTS_DIR'])
        for name in MODEL_RUNS:
            OmegaConf.load(root/name/'config.yml');state=torch.load(root/name/'model_best.pth',map_location='cpu',weights_only=True)
            state=state.get('model',state);t=[v for v in state.values() if torch.is_tensor(v)]
            assert t and all(torch.isfinite(v).all() for v in t);checked.append(dict(model=name,tensors=len(t)))
        result=dict(passed=True,native_frames=len(frames),models=checked,external_poses=len(poses),
            observed_frames=int(sum(observed)),missing_observations=ids[~np.asarray(observed)].tolist(),
            mesh_vertices=len(mesh.vertices),mesh_faces=len(mesh.faces),gpu_used=False)
        write(output/'CPU_CHECK.json',result);print(json.dumps(result),flush=True);return result
    install_rgb_mode(up);debug=output/'debug';debug.mkdir(exist_ok=True)
    est=up._pose_estimator(mesh,debug);transform=up._tensor_to_numpy(est.get_tf_to_centered_mesh())
    inverse=np.linalg.inv(transform);points=sample_vertices(est.mesh_ori.vertices);configs=meta['hybrid_config']
    @torch.no_grad()
    def track(frame,previous):
        rgb,depth,human,obj,k=frames[frame]
        est.pose_last=torch.as_tensor((previous@inverse)[None],device='cuda',dtype=torch.float32)
        return est.track_one(rgb,depth,k,iteration=5)
    @torch.no_grad()
    def measure(frame,pose):
        rgb,depth,human,obj,k=frames[frame]
        masks=up._render_candidate_masks(est,(np.asarray(pose,np.float32)@inverse)[None],k,obj.shape)
        metrics=up._selected_metric_values(up.measure_visible_silhouettes(masks,human,obj),0)
        geometry=consistency(points,pose,k,depth,obj&~human,support)
        return dict(metrics=metrics,geometry=geometry)
    def provider(frame,previous):
        return [dict(name=meta['rows'][i]['name'],pose_camera=poses[i],provenance=meta['rows'][i]['provenance'])
            for i in np.flatnonzero(pids==frame)]
    results={}
    for mode in ('tracker_only','hybrid'):
        torch.manual_seed(0);np.random.seed(0);folder=output/mode;folder.mkdir(exist_ok=True)
        def record(row):
            write(folder/f'{row["frame"]:06d}.json',row)
            write(output/'status.json',dict(stage=mode,frame=row['frame'],status=row['status'],elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(mode=mode,frame=row['frame'],status=row['status'],accepted=row['accepted'])),flush=True)
        poses_out,accepted,result=run_sequence(ids,ids.astype(float)/configs['nominal_fps'],np.asarray(observed),
            np.asarray(anchor['pose_cam']),track=track,measure=measure,propose=provider if mode=='hybrid' else None,
            config=HybridConfig(**configs),record=record)
        np.savez_compressed(folder/'poses.npz',source_frame_ids=ids,poses_camera=poses_out,accepted=accepted)
        write(folder/'RESULTS.json',result);results[mode]=result
    write(output/'status.json',dict(stage='complete',elapsed_seconds=time.monotonic()-start,
        hybrid_observed_coverage_complete=results['hybrid']['observed_coverage_complete'],
        cached_proposal_replay=True,new_diffusion_inference=False,reference_approved=False))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','proposals','metadata','seed','occlusion','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--cpu-check',action='store_true');args=vars(parser.parse_args());execute(**args)
