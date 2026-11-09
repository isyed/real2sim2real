"""Regenerate an explicitly mapped RGB depth window with a fixed centered crop.

This produces predicted depth, not accepted object motion. Floor fitting uses
only reviewed static pixels; original poses never select a depth hypothesis.
"""
import argparse,io,json,time
from pathlib import Path
import numpy as np
from .foreground_depth import crop_camera,floor_align


def compose(original,aligned_crop,valid_crop,box):
    """Preserve outside pixels exactly; invalid new pixels stay unobserved."""
    old,new,valid=np.asarray(original),np.asarray(aligned_crop),np.asarray(valid_crop)
    b=np.asarray(box)
    if (old.ndim!=2 or old.dtype.kind!='f' or not np.isfinite(old).all() or np.any(old<0)
        or b.shape!=(4,) or b.dtype.kind not in 'iu' or valid.dtype.kind!='b'):
        raise ValueError('Finite metric depth, integer native crop and boolean validity required')
    x0,y0,x1,y1=map(int,b);h,w=old.shape
    if not (0<=x0<x1<=w and 0<=y0<y1<=h) or new.shape!=(y1-y0,x1-x0) or valid.shape!=new.shape:
        raise ValueError('Depth proposal does not match the declared native crop')
    if not np.isfinite(new).all() or np.any(new<0) or np.any(valid&((new<=0)|(new>=65.535))):
        raise ValueError('Valid proposal pixels require finite positive metric depth')
    result=old.copy();result[y0:y1,x0:x1]=np.where(valid,new,0)
    return result


def restore_outside_pixels(original_mm,candidate_mm,box):
    """Restore native integers after a float-to-PNG codec round trip."""
    original,candidate=np.asarray(original_mm),np.asarray(candidate_mm);b=np.asarray(box)
    if (original.ndim!=2 or original.dtype!=np.uint16 or candidate.dtype!=np.uint16
        or candidate.shape!=original.shape or b.shape!=(4,) or b.dtype.kind not in 'iu'):
        raise ValueError('Matching native uint16 grids and an integer crop required')
    x0,y0,x1,y1=map(int,b);h,w=original.shape
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Crop exceeds native depth grid')
    result=original.copy();result[y0:y1,x0:x1]=candidate[y0:y1,x0:x1]
    return result


def preserve_native_outside(source,archive,box):
    """Finalize an OWNED derived archive, preserving original aligned pixels.

    The engine's legacy encoder truncates float millimetres. Restoring the
    untouched region from decoded integers avoids a one-millimetre loss there.
    Original depth is read only; derived raw/aligned remain an identity composite.
    """
    import h5py
    from PIL import Image
    from r2s2r_engines.cari4d.lib.cari4d.prep.mhr_depth_h5 import _encode_depth_png_uint16
    source,archive=Path(source),Path(archive)
    if source.samefile(archive):raise ValueError('Never modify the original depth archive or a hardlink to it')
    camera='front_stereo_camera_left';changed=0
    with h5py.File(source) as old,h5py.File(archive,'r+') as new:
        n=len(old['aligned/'+camera])
        if not old.attrs.get('complete') or not new.attrs.get('complete') or len(new['aligned/'+camera])!=n:
            raise ValueError('Complete matching native archives required')
        new.attrs['complete']=False
        for i in range(n):
            a=np.asarray(Image.open(io.BytesIO(old['aligned/'+camera][i].tobytes())),np.uint16)
            b=np.asarray(Image.open(io.BytesIO(new['aligned/'+camera][i].tobytes())),np.uint16)
            fixed=restore_outside_pixels(a,b,box);changed+=int(np.count_nonzero(fixed!=b));payload=_encode_depth_png_uint16(fixed)
            new['raw/'+camera][i]=payload;new['aligned/'+camera][i]=payload
            new['alignment/'+camera+'/valid_count'][i]=int(((fixed>0)&(fixed<65535)).sum())
        new.attrs['complete']=True
    return dict(checked_frames=n,outside_crop_exact=True,crop_payload_values_preserved=True,
        codec_roundtrip_pixels_restored=changed,source_untouched=True)


def run(request):
    import h5py,torch
    from PIL import Image
    from moge.model.v2 import MoGeModel
    from real2sim2real.artifacts import digest,write_json
    from r2s2r_engines.cari4d.lib.cari4d.prep.mhr_depth_h5 import MHRDepthH5Writer,DepthFrameRecord
    request=Path(request);spec=json.loads(request.read_text());export=Path(spec['export']);depth=Path(spec['depth']);output=Path(spec['output'])
    if output.exists():raise FileExistsError('Preserve previous depth-window outputs')
    if spec.get('clean_trajectory_used') is not False or not spec.get('floor_evidence'):
        raise ValueError('Explicit RGB-reviewed floor evidence and no clean trajectories required')
    if not np.isfinite(spec['max_seconds']) or spec['max_seconds']<=0:
        raise ValueError('Positive finite depth-window time budget required')
    info=json.loads((export/'wild_export.json').read_text());mapping=json.loads(Path(spec['mapping']).read_text())
    ids=np.asarray(mapping['source_frame_indices']);n=info['frame_count'];K=np.asarray(info['intrinsics'],float)
    if ids.shape!=(n,) or ids.dtype.kind not in 'iu' or np.any(np.diff(ids)!=1) or n<5:
        raise ValueError('An explicit forward native-frame window is required')
    if info['source_video_sha256']!=mapping['source_video_sha256']:raise ValueError('Source RGB identities differ')
    box=spec['crop_xyxy'];x0,y0,x1,y1=box;shape=(info['height'],info['width']);camera='front_stereo_camera_left'
    cropK,cropfov=crop_camera(K,box,shape);_,fullfov=crop_camera(K,[0,0,shape[1],shape[0]],shape)
    rectangles=spec['floor_rectangles_xyxy'];local_rects=[[a-x0,b-y0,c-x0,d-y0] for a,b,c,d in rectangles]
    if any(not (0<=a<c<=x1-x0 and 0<=b<d<=y1-y0) for a,b,c,d in local_rects):
        raise ValueError('Reviewed floor regions must lie inside the fixed crop')
    controls=spec['full_control_source_frames']
    if not set(controls)<=set(ids.tolist()):raise ValueError('Full-image controls must lie in the source window')
    plane=np.asarray(spec['plane_camera'],float);output.mkdir(parents=True)
    identity=dict(source_depth_sha256=digest(depth),source_mapping_sha256=digest(spec['mapping']),request_sha256=digest(request),
        checkpoint_sha256=digest(spec['checkpoint']),camera_crop_xyxy=box,source_frame_ids=ids.tolist(),poses_used=False)
    write_json(output/'identity.json',identity)
    model=MoGeModel.from_pretrained(spec['checkpoint']).to('cuda').eval();started=time.monotonic();rows=[]
    writer=MHRDepthH5Writer(output/'depth.h5',{camera:[f'{i:06d}' for i in range(n)]},
        alignment_method='rgb_centered_crop_static_floor_patch_v1',alignment_input_identity=identity,
        encoding_workers=2,max_pending_batches=2)
    def predict(picture,fov):
        tensor=torch.from_numpy(np.ascontiguousarray(picture)).permute(2,0,1).to('cuda').float()/255
        with torch.inference_mode():pred=model.infer(tensor,resolution_level=9,force_projection=True,apply_mask=True,use_fp16=True,fov_x=fov)
        z=pred['depth'].detach().cpu().numpy().astype(np.float32);valid=pred['mask'].detach().cpu().numpy().astype(bool)&np.isfinite(z)&(z>0)&(z<65.535)
        z[~valid]=0
        return z,valid
    try:
        with h5py.File(export/f'images/{camera}.h5') as rgb,h5py.File(export/f'object_masks/{camera}.h5') as obj,h5py.File(export/f'human_masks/{camera}.h5') as human,h5py.File(depth) as old:
            if not old.attrs.get('complete') or any(len(h['frames'])!=n for h in [rgb,obj,human]):raise ValueError('Incomplete native inputs')
            for i,source in enumerate(ids):
                if time.monotonic()-started>spec['max_seconds']:raise TimeoutError('Bounded depth-sequence time budget expired')
                image=np.asarray(Image.open(io.BytesIO(rgb['frames'][i].tobytes())).convert('RGB'));objects=obj['frames'][i]!=0;humans=human['frames'][i]!=0
                if image.shape[:2]!=shape or objects.shape!=shape or humans.shape!=shape:raise ValueError('Native image grids differ')
                z,valid=predict(image[y0:y1,x0:x1],cropfov);excluded=(objects|humans)[y0:y1,x0:x1]
                aligned,fit=floor_align(z,cropK,plane,local_rects,excluded)
                visible=(objects&~humans)[y0:y1,x0:x1];observed=int(visible.sum());covered=int((visible&valid).sum())
                row=dict(native_source_frame=int(source),tracker_local_frame=i,floor=fit,visible_object_pixels=observed,
                    valid_visible_depth_pixels=covered,object_depth_quantiles_m=np.quantile(aligned[visible&valid],[.1,.5,.9]).tolist() if covered else [],proposal_only=True)
                rows.append(row)
                if not fit['accepted'] or (observed>=100 and covered/observed<.95):
                    write_json(output/'quality_rejection.json',row);raise ValueError('Crop depth failed floor or object-coverage qualification')
                original=np.asarray(Image.open(io.BytesIO(old['aligned/'+camera][i].tobytes())),np.float32)/1000
                composite=compose(original,aligned,valid,box)
                if int(source) in controls:
                    full,fvalid=predict(image,fullfov);full_aligned,control=floor_align(full,K,plane,rectangles,objects|humans)
                    m=objects&~humans&fvalid;control['object_depth_quantiles_m']=np.quantile(full_aligned[m],[.1,.5,.9]).tolist() if m.any() else []
                    row['full_control']=control
                writer.submit_frames(camera,[DepthFrameRecord(i,composite,composite,1.,0.,int(((composite>0)&(composite<65.535)).sum()))])
                write_json(output/'progress.json',dict(completed=i+1,total=n,last=row,reference_poses_approved=False))
                print('CROP_DEPTH_FRAME',int(source),fit['heldout_p90_residual_m'],row['object_depth_quantiles_m'],flush=True)
            writer.mark_complete()
    finally:writer.close()
    pixel_receipt=preserve_native_outside(depth,output/'depth.h5',box)
    write_json(output/'native_pixel_receipt.json',pixel_receipt)
    result=dict(schema='real2sim2real.crop-depth-sequence/1',frames=n,rows=rows,
        **identity,depth_sha256=digest(output/'depth.h5'),outside_crop_preserved=True,
        invalid_new_pixels_filled_from_old=False,raw_equals_aligned_composite=True,
        raw_semantics='Derived RGB depth composite; immutable original model depth retained separately',
        source_only_prediction=True,metric_truth_known=False,all_floor_checks_passed=True,native_pixel_preservation=pixel_receipt,
        complete_reference_accepted=False,physics_validated=False,requires_pose_revalidation=True)
    write_json(output/'RESULTS.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);args=parser.parse_args();run(args.request)
