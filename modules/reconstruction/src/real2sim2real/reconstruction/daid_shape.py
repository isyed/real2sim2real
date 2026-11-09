"""Prepare a fixed metric mesh for the optional external recovery runtime.

The upstream implementation, environment and weights remain external. CPU
qualification checks real preprocessing and coordinate conventions; shape
encoding is a separate caller-supervised GPU stage, never a mesh replacement.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path


def run(checkout,config_dir,inputs,output,*,cpu_check):
    fork=Path(checkout)/'reconstruction/modules/Fast-SAM3D'
    sys.path[:0]=[str(fork),'/opt/vendor'];os.chdir(fork)
    import numpy as np,torch
    import track_object as tracker
    from omegaconf import OmegaConf
    from hydra.utils import instantiate,get_method
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from sam3d_objects.pipeline.inference_utils import normalize_mesh_verts
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    from .daid_inputs import shared_pointmap_dict
    from real2sim2real.reconstruction_repair.canonical_pose import encode,decode
    data,out=Path(inputs),Path(output);out.mkdir(parents=True,exist_ok=True)
    def write(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
    config=OmegaConf.load(str(Path(config_dir)/'pipeline.yaml'))
    config.workspace_dir=str(config_dir);config.ss_generator_config_path='ss_generator_faster.yaml'
    if cpu_check:
        assert not torch.cuda.is_available()
        z=np.load(data/'anchor_input.npz');pose=z['pose_camera'];K=z['K']
        mesh=tracker.trimesh.load(data/'shared_mesh.obj',force='mesh',process=False)
        Q=tracker._R_YUP_TO_ZUP_RENDER.astype(float);D=np.diag([-1.,-1.,1.])
        canonical,norm,center=normalize_mesh_verts(np.asarray(mesh.vertices)@Q)
        tracker.trimesh.Trimesh(canonical@Q.T,mesh.faces,process=False).export(out/'canonical_mesh.obj')
        a=encode(pose,Q,norm,center,D)
        assert np.allclose(decode(**a,axes=Q,normalization=norm,center=center,camera_axes=D),pose,atol=2e-6)
        raw=dict(rotation=torch.tensor(Rotation.from_matrix(a['rotation']).as_quat(scalar_first=True)[None],dtype=torch.float32),
                 translation=torch.tensor(a['translation'][None],dtype=torch.float32),scale=torch.tensor(a['scale'][None],dtype=torch.float32))
        tfm=tracker.compose_transform(raw['scale'],tracker.quaternion_to_matrix(raw['rotation']),raw['translation'])
        actual=tfm.transform_points(torch.tensor(canonical[None],dtype=torch.float32))[0].numpy()@D
        expected=np.asarray(mesh.vertices)@pose[:3,:3].T+pose[:3,3]
        error=float(np.max(abs(actual-expected)));assert error<3e-6,error
        occupancy,*_=tracker.voxelize(str(data/'shared_mesh.obj'));other,*_=tracker.voxelize(str(out/'canonical_mesh.obj'))
        assert torch.equal(occupancy,other)
        write('anchor_layout.json',dict(K=K.tolist(),Q=Q.tolist(),normalization=float(norm),center_z=center.tolist(),physical_size_changed=False))
        pre=instantiate(config.ss_preprocessor);dummy=InferencePipelinePointMap.__new__(InferencePipelinePointMap);dummy.device=torch.device('cpu')
        frames=json.loads((data/'frames.json').read_text());checked=[]
        for f in frames:
            rgb=np.asarray(Image.open(data/f'{f:06d}.png').convert('RGB'));mask=np.asarray(Image.open(data/f'{f:06d}_mask.png'))>0
            depth=np.load(data/f'{f:06d}_original.npz');assert int(depth['source_frame'])==f and np.allclose(depth['K'],K)
            if not mask.any():continue
            rgba=np.concatenate([rgb,(mask*255).astype('uint8')[...,None]],-1)
            points=shared_pointmap_dict(rgb,depth['depth_m'],K,torch.device('cpu'))
            item=dummy.preprocess_image(rgba,pre,pointmap=points['pointmap'])
            latent=tracker._prepare_pose_target_latent(raw,item,torch.device('cpu'))
            assert all(torch.isfinite(v).all() for v in latent.values());checked.append(f)
        names=set()
        def collect(x):
            if isinstance(x,dict):
                if '_target_' in x:names.add(x['_target_'])
                for v in x.values():collect(v)
            elif isinstance(x,list):
                for v in x:collect(v)
        for file in ('pipeline.yaml','ss_generator_faster.yaml'):
            collect(OmegaConf.to_container(OmegaConf.load(str(Path(config_dir)/file)),resolve=False))
        for name in names:assert callable(get_method(name))
        write('CPU_CHECK.json',dict(passed=True,native_frames=len(frames),observed_frames=checked,import_targets=len(names),
            native_pose_preprocessing=True,metric_roundtrip_error_m=error,voxel_occupancy_identical=True,
            shape_encoding_gpu_pending=True,gpu_used=False))
        print('Native shape/pose preprocessing passed',len(checked),flush=True)
    else:
        torch.cuda.set_per_process_memory_fraction(.88);torch.manual_seed(42)
        print('Loading external shape encoder',flush=True);start=time.monotonic()
        pipeline=tracker.Inference(config,compile=False)._pipeline
        occupancy,*_=tracker.voxelize(str(data/'canonical_mesh.obj'))
        print('Encoding fixed mesh occupancy',flush=True)
        z,_=tracker.encode_target_occupancy(pipeline,occupancy,torch.device('cuda'),opt_steps=1000)
        assert torch.isfinite(z).all();torch.save(z.detach().cpu(),out/'shape_latent.pt')
        write('SHAPE_CACHE.json',dict(complete=True,canonical_mesh_sha256=hashlib.sha256((data/'canonical_mesh.obj').read_bytes()).hexdigest(),
            latent_sha256=hashlib.sha256((out/'shape_latent.pt').read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-start,
            physical_mesh_changed=False))
        print('Shape cache complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkout','config-dir','inputs','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--cpu-check',action='store_true');run(**vars(p.parse_args()))
