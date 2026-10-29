"""Recover monocular object tracking from observed masks and aligned depth.

Preserves the mesh/scale and original FoundationPose results. Fits rigid pose
sequentially with antialiased silhouettes, human occlusion, and weak depth and
rotation continuity constraints. Every frame remains subject to visual QA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import Utils
from prep.mhr_export_utils import camera_calibration,load_edex,mask_input_identity
from prep.mhr_depth_h5 import depth_h5_file_identity
from prep.run_foundationpose_mhr_export import _foundationpose_frame_iterator,load_export_object_mesh


def rot6(x):
    a=F.normalize(x[:,:3],dim=-1);b=F.normalize(x[:,3:]-(a*x[:,3:]).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack([a,b,torch.cross(a,b,dim=-1)],-1)


def local_rotation(x):
    """Bound incremental rotation to 15 degrees; zero has a finite derivative."""
    norm=torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-8)
    omega=x*(torch.tanh(norm)/norm)*np.deg2rad(15)
    a,b,c=omega.unbind(-1);z=torch.zeros_like(a)
    skew=torch.stack([z,-c,b,c,z,-a,-b,a,z],-1).reshape(-1,3,3)
    return torch.matrix_exp(skew)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export-seq','depth-root','input-poses','out-file'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=45)
    p.add_argument('--bounded-motion',action='store_true',help='New candidate branch with local rotation/translation trust regions')
    p.add_argument('--start',type=int,default=0,help='First processed frame; preserve baseline outside the interval')
    p.add_argument('--end',type=int,default=None,help='Exclusive interval end; default is the full sequence')
    a=p.parse_args()
    if a.out_file.exists():raise FileExistsError(a.out_file)
    a.out_file.parent.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(a.input_poses);names=list(bundle['frames']);K,w2c=camera_calibration(load_edex(a.export_seq),0)
    end=len(names) if a.end is None else a.end
    if not 0<=a.start<end<=len(names):raise ValueError('Invalid recovery interval')
    if not np.allclose(w2c,np.eye(4),atol=1e-6):raise ValueError('Requires camera-space wild export')
    mesh=load_export_object_mesh(a.export_seq)
    if len(mesh.faces)>10000:mesh=mesh.simplify_quadric_decimation(face_count=10000)
    v=torch.tensor(np.asarray(mesh.vertices),device='cuda',dtype=torch.float32)
    tensors={'faces':torch.tensor(np.asarray(mesh.faces),device='cuda',dtype=torch.int32),
             'vertex_color':torch.ones((len(mesh.vertices),3),device='cuda')}
    ctx=dr.RasterizeCudaContext();baseline=np.asarray(bundle['obj_pose_world']);poses=[];rows=[]
    signature={'poses_sha256':hashlib.sha256(a.input_poses.read_bytes()).hexdigest(),
               'mesh_sha256':hashlib.sha256((a.export_seq/'object_mesh/output_aligned.glb').read_bytes()).hexdigest(),
               'depth_identity':depth_h5_file_identity(a.depth_root),
               'masks':{kind:mask_input_identity(a.export_seq,kind,0) for kind in ('human','object')},
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'renderer_sha256':hashlib.sha256(Path(Utils.__file__).read_bytes()).hexdigest(),
               'steps':a.steps,'bounded_motion':a.bounded_motion,'start':a.start,'end':end,'schema':'r2s2r_engines.observed_tracking.v3'}
    checkpoint=a.out_file.with_suffix('.resume.pkl')
    if checkpoint.exists():
        state=joblib.load(checkpoint)
        if state['signature']!=signature:raise ValueError('Tracking recovery identity differs')
        poses=state['poses'];rows=state['rows']
    resume_count=len(poses)
    for i,(_,rgb,depth,human,mask,K_frame) in enumerate(_foundationpose_frame_iterator(a.export_seq,a.depth_root,0,names,K,True)):
        if i<a.start+resume_count:continue
        if i>=end:break
        previous=baseline[a.start] if not poses else np.asarray(poses[-1])
        yy,xx=np.nonzero(mask)
        if len(xx)<30:
            poses.append(previous.copy());rows.append({'frame':i,'iou':None,'visibility':None,'status':'HELD_INSUFFICIENT_MASK'})
            continue
        side=max(32,float(max(np.ptp(xx),np.ptp(yy))*1.65));cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2
        x0=cx-side/2;y0=cy-side/2;size=192
        affine=np.array([[size/side,0,-x0*size/side],[0,size/side,-y0*size/side]],np.float32)
        target_np=cv2.warpAffine(mask.astype(np.float32),affine,(size,size),flags=cv2.INTER_LINEAR)
        human_np=cv2.warpAffine(human.astype(np.float32),affine,(size,size),flags=cv2.INTER_NEAREST)
        target=torch.tensor(target_np,device='cuda');keep=torch.tensor((human_np<.5)|(target_np>.5),device='cuda')
        Kr=K_frame.copy();Kr[0]*=size/side;Kr[1]*=size/side;Kr[0,2]-=x0*size/side;Kr[1,2]-=y0*size/side
        candidates=[]
        for source in (previous,baseline[i]):
            candidate=source.copy();points=np.asarray(mesh.vertices)@candidate[:3,:3].T+candidate[:3,3]
            uv=points@K_frame.T;uv=uv[:,:2]/np.maximum(uv[:,2:],.01);pc=(uv.min(0)+uv.max(0))/2
            z=float(np.median(points[:,2]));candidate[0,3]+=(cx-pc[0])*z/K_frame[0,0];candidate[1,3]+=(cy-pc[1])*z/K_frame[1,1]
            candidates.append(candidate)
        candidates.append(baseline[i].copy())
        r0=np.asarray([x[:3,:3] for x in candidates]);t0=np.asarray([x[:3,3] for x in candidates])
        r6=torch.nn.Parameter(torch.tensor(np.concatenate([r0[:,:,0],r0[:,:,1]],1),device='cuda',dtype=torch.float32))
        t=torch.nn.Parameter(torch.tensor(t0,device='cuda',dtype=torch.float32))
        optimizer=torch.optim.Adam([{'params':[r6],'lr':.012},{'params':[t],'lr':.002}])
        previous_R=torch.tensor(previous[:3,:3],device='cuda',dtype=torch.float32)
        if a.bounded_motion:
            # Do not allow a stale baseline candidate to teleport onto a
            # silhouette-equivalent but physically unrelated pose.
            starts=np.vstack([np.zeros((1,3)),np.eye(3)*.35,-np.eye(3)*.35])
            delta=torch.nn.Parameter(torch.tensor(starts,device='cuda',dtype=torch.float32))
            previous_t=torch.tensor(previous[:3,3],device='cuda',dtype=torch.float32)
            shift=np.clip((candidates[0][:3,3]-previous[:3,3])/.045,-.95,.95)
            translation_delta=torch.nn.Parameter(torch.tensor(np.repeat(np.arctanh(shift)[None],7,0),device='cuda',dtype=torch.float32))
            optimizer=torch.optim.Adam([{'params':[delta],'lr':.04},{'params':[translation_delta],'lr':.04}])
        valid=depth[mask&~human&np.isfinite(depth)&(depth>.01)]
        target_depth=float(np.median(valid)) if len(valid)>20 else float(previous[2,3])
        best=None;gradient_norm=None
        for step in range(a.steps+1):
            if a.bounded_motion:
                r=local_rotation(delta)@previous_R
                t=previous_t+.045*torch.tanh(translation_delta)
            else:r=rot6(r6)
            points=v[None]@r.transpose(1,2)+t[:,None]
            color,_,_=Utils.nvdiff_color_depth_render(np.repeat(Kr[None],len(r),0),ctx,tensors,(size,size),points,antialias=True)
            full=color.mean(-1);pred=full*keep
            intersection=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1)
            iou=intersection/union
            z=torch.quantile(points[:,:,2],.2,dim=1)
            depth_residual=(z-target_depth)/max(.1,.1*target_depth)
            depth_error=F.smooth_l1_loss(depth_residual,torch.zeros_like(depth_residual),reduction='none') if a.bounded_motion else depth_residual.square()
            rotation_error=(r-previous_R).square().mean((1,2))
            loss=1-iou+.03*depth_error+.015*rotation_error
            score,idx=loss.min(0)
            if best is None or float(score)<best['loss']:
                pose=np.eye(4,dtype=np.float32);pose[:3,:3]=r[idx].detach().cpu().numpy();pose[:3,3]=t[idx].detach().cpu().numpy()
                best={'loss':float(score),'pose':pose,'iou':float(iou[idx]),'visibility':float(pred[idx].sum()/full[idx].sum().clamp_min(1))}
            if step==a.steps:break
            optimizer.zero_grad();loss.sum().backward()
            if gradient_norm is None:gradient_norm=float((translation_delta.grad if a.bounded_motion else t.grad).norm())
            optimizer.step()
        poses.append(best.pop('pose'));rows.append({'frame':i,**best,'translation_gradient_norm':gradient_norm,'status':'CANDIDATE'})
        if (i+1)%10==0 or i+1==end:
            with checkpoint.with_suffix('.tmp').open('wb') as f:pickle.dump({'signature':signature,'poses':poses,'rows':rows},f)
            checkpoint.with_suffix('.tmp').replace(checkpoint)
            print('OBSERVED_TRACK '+json.dumps({'frame':i,'total':len(names),'iou':rows[-1]['iou'],'gradient_norm':gradient_norm}),flush=True)
    partial=np.asarray(poses,dtype=np.float32)
    poses=baseline.copy();poses[a.start:end]=partial
    ious=np.asarray(bundle['fp_selected_iou']).copy();visibility=np.asarray(bundle['fp_selected_visibility']).copy();reliable=np.asarray(bundle['fp_reliable']).copy()
    ious[a.start:end]=np.array([r['iou'] if r['iou'] is not None else np.nan for r in rows],np.float32)
    visibility[a.start:end]=np.array([r['visibility'] if r['visibility'] is not None else np.nan for r in rows],np.float32)
    reliable[a.start:end]=np.array([r['iou'] is not None and r['iou']>=.5 for r in rows])
    result={**bundle,'obj_pose_world':poses,'obj_rot_init':poses[:,:3,:3],'obj_t_init':poses[:,:3,3],
            'fp_poses_cam':poses[:,None], 'fp_selected_iou':ious,
            'fp_selected_visibility':visibility,
            'fp_reliable':reliable,
            'metadata':{**bundle['metadata'],'observed_tracking':signature,'source':'observed_mask_depth_tracking_correction',
                        'diagnostic_caveat':'Pose, IoU, visibility, reliable flags replaced; other fp selection counters describe preserved baseline only'}}
    with a.out_file.with_suffix('.tmp').open('wb') as f:pickle.dump(result,f)
    a.out_file.with_suffix('.tmp').replace(a.out_file)
    a.out_file.with_suffix('.report.json').write_text(json.dumps({'signature':signature,'frames':rows,'status':'REQUIRES_FULL_SEQUENCE_REVIEW'},indent=2)+'\n')


if __name__=='__main__':main()
