"""Validate/refine explicit tracking proposals in anchored temporal windows.

Source/local frame identities survive reverse-time tracking. Only independently
passed windows enter the accepted ledger; failed and unprocessed frames remain
unaccepted. This diagnostic composition does not certify metric size or approve
a complete reconstructed reference.
"""
from dataclasses import asdict
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np

from real2sim2real.artifacts import digest,write_json
from .temporal_pose import TemporalPoseConfig,acceptance,measurements,refine_window


def source_order(indices):
    ids=np.asarray(indices)
    if (ids.ndim!=1 or len(ids)<5 or ids.dtype.kind not in 'iu' or np.any(ids<0)
            or not (np.all(np.diff(ids.astype(np.int64))==1) or np.all(np.diff(ids.astype(np.int64))==-1))):
        raise ValueError('Explicit consecutive forward/backward source IDs required')
    return np.argsort(ids)


def bounded_windows(count,size):
    if type(count) is not int or type(size) is not int or count<5 or size<5:
        raise ValueError('At least five frames and a five-frame window required')
    start=0
    while start<count-1:
        end=min(count,start+size)
        if count-end in (1,2,3):end=count
        yield start,end
        if end==count:break
        start=end-1


def refine(export,depth,proposals,mapping,support,output,*,config=TemporalPoseConfig(),
           window_frames=31,max_total_seconds=1800.):
    import h5py,trimesh
    from PIL import Image
    export,depth,proposals,mapping,support,output=map(Path,(export,depth,proposals,mapping,support,output))
    config.validate()
    if output.exists():raise FileExistsError(output)
    if not np.isfinite(max_total_seconds) or max_total_seconds<=0:raise ValueError('Finite CPU budget required')
    with proposals.open('rb') as stream:raw=pickle.load(stream)
    receipt=json.loads(proposals.with_suffix('.proposal_receipt.json').read_text())
    meta=raw['metadata']
    if (receipt.get('output_sha256')!=digest(proposals) or receipt.get('proposal_only') is not True
            or meta.get('proposal_only') is not True or meta.get('gt_object_pose_available') is not False):
        raise ValueError('Hash-bound prediction-only tracking proposals required')
    info=json.loads((export/'wild_export.json').read_text());source=json.loads(mapping.read_text())
    if (source.get('source_video_sha256')!=info.get('source_video_sha256')
            or source.get('parent_artifact_sha256')!=info.get('parent_artifact_sha256')):
        raise ValueError('Mapping and tracking export source identities differ')
    ids=np.asarray(source['source_frame_indices'],np.int64);n=len(ids);source_order(ids)
    poses=np.asarray(raw['fp_poses_cam'])[:,0].astype(float)
    if poses.shape!=(n,4,4) or info['frame_count']!=n or not np.isfinite(poses).all():
        raise ValueError('Proposal/mapping coverage mismatch')
    if not np.array_equal(np.asarray(raw['frames']).astype(int),np.arange(n)):
        raise ValueError('Proposal local frame names differ from the declared mapping')
    K=np.asarray(info['intrinsics'],float)
    input_hashes=dict(depth=digest(depth),mapping=digest(mapping),support=digest(support),
        export_metadata=digest(export/'wild_export.json'),mesh=digest(export/'object_mesh/output_aligned.glb'),
        object_masks=digest(export/'object_masks/front_stereo_camera_left.h5'),
        human_masks=digest(export/'human_masks/front_stereo_camera_left.h5'))
    support=json.loads(support.read_text())
    if support.get('ground_truth_used') is not False or support.get('coordinate_frame')!='camera':
        raise ValueError('Predicted camera-frame support required')
    mesh=trimesh.load(export/'object_mesh/output_aligned.glb',force='mesh',process=False)
    output.mkdir(parents=True);started=time.monotonic();accepted=np.zeros(n,bool)
    # The caller's source-video frame interval is fixed in the input receipt.
    fps=float(source['source_fps'])
    if not np.isfinite(fps) or fps<=0:raise ValueError('Explicit positive source frame rate required')
    reports=[]
    with h5py.File(depth) as depths,h5py.File(export/'object_masks/front_stereo_camera_left.h5') as obj,h5py.File(export/'human_masks/front_stereo_camera_left.h5') as human:
        for start,end in bounded_windows(n,window_frames):
            if time.monotonic()-started>=max_total_seconds:
                reports.append(dict(local_range=[start,end-1],accepted=False,reasons=['cpu_batch_budget']));break
            local=np.arange(start,end);order=source_order(ids[local]);idx=local[order]
            window=poses[idx].copy();locked=idx==start
            data=dict(vertices=mesh.vertices,faces=mesh.faces,poses_camera=window,
                intrinsics=np.tile(K,(len(idx),1,1)),frame_ids=ids[idx],timestamps_s=ids[idx]/fps,
                object_mask=np.stack([obj['frames'][int(i)]!=0 for i in idx]),
                occluder_mask=np.stack([human['frames'][int(i)]!=0 for i in idx]),
                depth_m=np.stack([np.asarray(Image.open(io.BytesIO(depths['aligned/front_stereo_camera_left'][int(i)].tobytes())),np.float32)/1000 for i in idx]),
                fit=np.arange(len(idx))%2==0,locked=locked,
                support_plane_camera=np.tile(support['plane_camera'],(len(idx),1)),
                support_center_camera=np.tile(support['center_camera_m'],(len(idx),1)),
                support_radius_m=np.full(len(idx),support['radius_m']))
            folder=output/f'window_{int(ids[idx[0]]):06d}_{int(ids[idx[-1]]):06d}';folder.mkdir()
            try:
                before=measurements(data,window,config)
                baseline=acceptance(data,before,before,window,window,config)
                needed=[r for r in baseline['reasons'] if r!='no_temporal_benefit']
                if not needed:
                    candidate=window;report=dict(decision=dict(accepted=True,reasons=[]),
                        before=before,after=before,action='retain_already_valid_proposals',config=asdict(config))
                else:
                    candidate,report=refine_window(data,config);report['action']='joint_temporal_refinement'
                report.update(source_frame_ids=ids[idx].tolist(),diagnostic_only=True,
                    metric_size_certified=False,full_reference_accepted=False,physics_validated=False)
                write_json(folder/'report.json',report)
                row=dict(source_range=[int(ids[idx[0]]),int(ids[idx[-1]])],local_range=[start,end-1],
                    accepted=report['decision']['accepted'],reasons=report['decision']['reasons'],action=report['action'])
                reports.append(row)
                if not row['accepted']:break
                poses[idx]=candidate;accepted[idx]=True
                np.savez_compressed(folder/'accepted_poses.npz',source_frame_ids=ids[idx],poses_camera=candidate)
            except (ValueError,RuntimeError) as error:
                row=dict(local_range=[start,end-1],accepted=False,reasons=[str(error)])
                reports.append(row);write_json(folder/'failure.json',row);break
            write_json(output/'progress.json',dict(accepted_frames=int(accepted.sum()),total=n,windows=reports))
    np.savez_compressed(output/'poses_with_acceptance.npz',source_frame_ids=ids,poses_camera=poses,accepted=accepted)
    result=dict(schema='real2sim2real.tracking-window-sequence/1',windows=reports,
        accepted_source_frames=ids[accepted].tolist(),accepted_frames=int(accepted.sum()),total=n,
        complete=bool(accepted.all()),config=asdict(config),seconds=time.monotonic()-started,
        source_mapping_sha256=digest(mapping),proposal_sha256=digest(proposals),
        input_sha256=input_hashes,
        full_reference_accepted=False,metric_size_certified=False,physics_validated=False,
        limitation='Native mask/depth and temporal development checks; no whole-reference seam or physical validation.')
    write_json(output/'RESULTS.json',result)
    return result
