"""Optional Do as I Do recovery adapter; upstream checkout and weights are external.

Run this module inside the upstream-compatible environment, under a caller-owned
GPU thermal/time supervisor. It preserves the fixed mesh/camera, saves all sampled
poses, and never accepts or promotes its own output. No upstream code is vendored.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
from functools import partial


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_request(path, *, require_accepted=False):
    import numpy as np
    from .recovery_quality import rigid
    s=json.loads(Path(path).read_text())
    for key in ('checkout','config_dir','input_dir','output_dir','accepted_state_path'):
        if not Path(s[key]).is_absolute():raise ValueError('Explicit absolute runtime paths required: '+key)
    data=Path(s['input_dir']);frame=s['source_frame'];variant=s['depth_hypothesis']
    if (type(frame)!=int or type(s['previous_frame'])!=int or not 1<=frame-s['previous_frame']<=3
        or not isinstance(variant,str) or not variant or not all(c.isalnum() or c=='_' for c in variant)
        or not rigid(np.asarray(s['previous_pose_camera']))):raise ValueError('Explicit accepted state and bounded source-frame gap required')
    state=Path(s['accepted_state_path'])
    if sha(state)!=s['accepted_state_sha256']:raise ValueError('Accepted-state receipt changed')
    receipt=json.loads(state.read_text())
    if receipt['source_frame']!=s['previous_frame'] or not np.array_equal(receipt['pose_camera'],s['previous_pose_camera']):
        raise ValueError('Conditioning pose differs from accepted state')
    if require_accepted and (receipt.get('accepted') is not True or not receipt.get('validation_provenance')):
        raise ValueError('Fresh GPU inference requires an independently accepted state receipt')
    required=['canonical_mesh.obj','shared_mesh.obj','shape_latent.pt','anchor_layout.json',
              f'{frame:06d}.png',f'{frame:06d}_mask.png',f'{frame:06d}_{variant}.npz']
    if set(s['input_sha256'])!=set(required):raise ValueError('Complete input hashes required')
    for name in required:
        if sha(data/name)!=s['input_sha256'][name]:raise ValueError('Input changed: '+name)
    fork=Path(s['checkout'])/'reconstruction/modules/Fast-SAM3D'
    required_upstream=['track_object.py','sam3d_objects/pipeline/inference_pipeline_pointmap.py']
    if set(s['upstream_sha256'])!=set(required_upstream):raise ValueError('Pinned upstream entry points required')
    for name in required_upstream:
        if sha(fork/name)!=s['upstream_sha256'][name]:raise ValueError('Upstream version changed: '+name)
    return s


def run(request, *, cpu_check):
    s=validate_request(request,require_accepted=not cpu_check)
    FORK=Path(s['checkout'])/'reconstruction/modules/Fast-SAM3D'
    sys.path.insert(0,str(FORK))
    if s.get('vendor_dir'):sys.path.insert(0,str(Path(s['vendor_dir'])))
    os.chdir(FORK)
    DATA=Path(s['input_dir']);OUT=Path(s['output_dir']);CONFIG=Path(s['config_dir'])
    OUT.mkdir(parents=True,exist_ok=True)
    def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    import numpy as np,torch
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from omegaconf import OmegaConf
    from hydra.utils import instantiate,get_method
    import track_object as tracker
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    from real2sim2real.reconstruction_repair.canonical_pose import encode,decode
    from real2sim2real.reconstruction_repair.fixed_scale_pose import restore_fixed_scale_projection
    from .daid_inputs import shared_pointmap_dict
    from .daid_render import native_binned_iou,bin_parameters
    from .daid_progress import observe_completed_calls
    layout=json.loads((DATA/'anchor_layout.json').read_text())
    frame=s['source_frame'];variant=s['depth_hypothesis'];K=np.asarray(layout['K']);Q=np.asarray(layout['Q']);D=np.diag([-1.,-1.,1.])
    history=np.asarray(s['previous_pose_camera']);canon=dict(axes=Q,normalization=layout['normalization'],center=np.asarray(layout['center_z']),camera_axes=D)
    encoded=encode(history,**canon);assert np.allclose(decode(**encoded,**canon),history,atol=2e-6)
    pose=dict(rotation=torch.tensor(Rotation.from_matrix(encoded['rotation']).as_quat(scalar_first=True)[None],dtype=torch.float32),
              translation=torch.tensor(encoded['translation'][None],dtype=torch.float32),scale=torch.tensor(encoded['scale'][None],dtype=torch.float32))
    rgb=np.asarray(Image.open(DATA/f'{frame:06d}.png').convert('RGB'));mask=np.asarray(Image.open(DATA/f'{frame:06d}_mask.png'))>0
    z=np.load(DATA/f'{frame:06d}_{variant}.npz');depth=z['depth_m'];assert np.allclose(z['K'],K) and int(z['source_frame'])==frame and mask.any()
    rgba=np.concatenate([rgb,(mask*255).astype('uint8')[...,None]],-1)
    config=OmegaConf.load(str(CONFIG/'pipeline.yaml'));config.workspace_dir=str(CONFIG);config.ss_generator_config_path='ss_generator_faster.yaml'
    if cpu_check:
        assert not torch.cuda.is_available();device=torch.device('cpu');pre=instantiate(config.ss_preprocessor)
        dummy=InferencePipelinePointMap.__new__(InferencePipelinePointMap);dummy.device=device
        points=shared_pointmap_dict(rgb,depth,K,device);inputs=dummy.preprocess_image(rgba,pre,pointmap=points['pointmap'])
        latent=tracker._prepare_pose_target_latent(pose,inputs,device);assert latent and all(torch.isfinite(x).all() for x in latent.values())
        mesh=tracker.trimesh.load(DATA/'canonical_mesh.obj',force='mesh',process=False)
        original=tracker.trimesh.load(DATA/'shared_mesh.obj',force='mesh',process=False)
        tfm=tracker.compose_transform(pose['scale'],tracker.quaternion_to_matrix(pose['rotation']),pose['translation'])
        actual=tfm.transform_points(torch.tensor((np.asarray(mesh.vertices)@Q)[None],dtype=torch.float32))[0].numpy()@D
        expected=np.asarray(original.vertices)@history[:3,:3].T+history[:3,3]
        error=float(np.max(abs(actual-expected)));assert error<3e-6,error
        names=set()
        def collect(x):
            if isinstance(x,dict):
                if '_target_' in x:names.add(x['_target_'])
                for v in x.values():collect(v)
            elif isinstance(x,list):
                for v in x:collect(v)
        for name in ('pipeline.yaml','ss_generator_faster.yaml'):collect(OmegaConf.to_container(OmegaConf.load(str(CONFIG/name)),resolve=False))
        for name in names:assert callable(get_method(name))
        cached=torch.load(DATA/'shape_latent.pt',weights_only=True,map_location='cpu');assert torch.isfinite(cached).all()
        write('CPU_CHECK.json',dict(passed=True,frame=frame,depth_hypothesis=variant,actual_pose_target_latent_checked=True,
            metric_roundtrip_error_m=error,import_targets=len(names),mesh_faces=len(mesh.faces),shape_cache_finite=True,
            native_binning=bin_parameters(rgb.shape[1],rgb.shape[0],len(mesh.faces)),gpu_used=False));return
    started=time.monotonic();torch.cuda.set_per_process_memory_fraction(.88);device=torch.device('cuda');fixed=pose['scale'].clone()
    mesh=tracker.trimesh.load(DATA/'canonical_mesh.obj',force='mesh',process=False);render=tracker._build_renderable_mesh(mesh,device)
    normalized=torch.tensor(K,dtype=torch.float32);normalized[0]/=rgb.shape[1];normalized[1]/=rgb.shape[0]
    native=tracker.compute_render_iou;captured=[];original_render=tracker._render_silhouette
    def capture(*args,**kwargs):
        a=original_render(*args,**kwargs);captured.append(a>.5);return a
    tracker._render_silhouette=capture
    a=native(render,pose,normalized,mask,rgb.shape[1],rgb.shape[0],device)
    b=native_binned_iou(tracker,render,pose,normalized,mask,rgb.shape[1],rgb.shape[0],device,full_scorer=native)
    disagreement=float(np.count_nonzero(captured[0]!=captured[1])/max(1,np.count_nonzero(captured[0]|captured[1])))
    write('RENDERER_PARITY.json',dict(native_iou=a,binned_iou=b,disagreement=disagreement));assert abs(a-b)<=1e-4 and disagreement<=1e-4
    tracker._render_silhouette=original_render;tracker.compute_render_iou=partial(native_binned_iou,tracker,full_scorer=native)
    pipeline=tracker.Inference(config,compile=False)._pipeline
    if pipeline.layout_post_optimization_method is None:pipeline.layout_post_optimization_method=tracker.layout_post_optimization
    def progress(count):
        torch.cuda.synchronize();print(json.dumps(dict(completed_sampling_calls=count,elapsed_seconds=time.monotonic()-started)),flush=True)
    observe_completed_calls(pipeline.models['ss_generator'],'_generate_dynamics',progress,interval=25)
    occupancy,*_=tracker.voxelize(str(DATA/'canonical_mesh.obj'));latent=torch.load(DATA/'shape_latent.pt',weights_only=True,map_location=device)
    pipeline.compute_pointmap=lambda image,pointmap=None:shared_pointmap_dict(image[...,:3],depth,K,device)
    native_post=pipeline.run_post_optimization;post_receipts=[]
    def post(*args,**kwargs):
        r=native_post(*args,**kwargs);target=kwargs.get('fixed_scale')
        if target is not None:
            R=tracker.quaternion_to_matrix(r['rotation'].detach().reshape(1,4)).cpu().numpy()[0].T
            corrected=restore_fixed_scale_projection(R,r['translation'].detach().cpu().numpy().reshape(3),r['scale'].detach().cpu().numpy().reshape(3),target.detach().cpu().numpy().reshape(3))
            r['translation']=torch.as_tensor(corrected['translation'],dtype=r['translation'].dtype,device=r['translation'].device).reshape_as(r['translation'])
            r['scale']=target.to(r['scale']).reshape_as(r['scale']).clone();post_receipts.append(dict(ratio=corrected['scale_ratio']))
        return r
    pipeline.run_post_optimization=post
    result=tracker.guided_predict_pose(pipeline,occupancy,rgba,latent.view(1,8,-1).permute(0,2,1).contiguous(),device,
        guidance_strength=1.,pose_guidance_strength=.5,pose_target=pose,seed=42,post_optimize=True,init_trimesh=mesh,
        fixed_scale=fixed,num_pose_samples=25,batch_chunk_size=4,scoring_metric='render_iou',render_mesh=render,
        gt_mask=mask,image_hw=mask.shape,enable_shape_icp=False,pose_selection='cluster',cluster_dist_thresh=.3,
        cluster_min_size=3,cluster_w_rot=1.5,ss_inference_steps=25)
    assert len(result['all_samples'])==25
    def camera(p):
        R=tracker.quaternion_to_matrix(p['rotation'].detach().float().reshape(1,4)).cpu().numpy()[0]
        return decode(R,p['translation'].detach().cpu().numpy(),p['scale'].detach().cpu().numpy(),**canon)
    poses=[camera(p) for p in result['all_samples']];names=[f'raw_sample_{i:02d}' for i in range(25)]
    if all('post_opt_'+k in result for k in pose):
        p={k:result['post_opt_'+k] for k in pose}
        if all(torch.isfinite(x).all() for x in p.values()):poses.append(camera(p));names.append('refined_cluster_winner')
    path=OUT/'proposals.npz';np.savez_compressed(path,poses_camera=np.asarray(poses,np.float32))
    write('proposals.json',dict(source_frame=frame,previous_frame=s['previous_frame'],conditioning_pose_camera=history.tolist(),
        source_video_sha256=s['source_video_sha256'],mesh_sha256=s['mesh_sha256'],depth_hypothesis=variant,
        depth_sha256=sha(DATA/f'{frame:06d}_{variant}.npz'),poses_sha256=sha(path),names=names,
        raw_candidates_retained=25,post_receipts=post_receipts,elapsed_seconds=time.monotonic()-started,
        selection_not_accepted=True,independent_native_checks_required=True,new_diffusion_inference=True))
    print('Fresh recovery proposals saved',frame,variant,len(poses),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--cpu-check',action='store_true')
    mode.add_argument('--run-gpu',action='store_true',help='Caller must enforce GPU thermal, memory, timeout and cleanup limits')
    a=p.parse_args();run(a.request,cpu_check=a.cpu_check)
