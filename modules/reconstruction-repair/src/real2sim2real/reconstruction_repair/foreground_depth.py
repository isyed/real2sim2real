"""Independent RGB depth proposals for small foreground objects.

Context crops preserve calibrated camera rays and adjust the cropped FOV.
A reviewed static floor fixes
scale using pixels outside object/person masks; held-out floor pixels check the
fit. Predictions are proposals, not observed truth or approved motion references.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np


def crop_camera(K, box, shape):
    K=np.asarray(K,float);box=np.asarray(box)
    if (K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0
            or box.shape!=(4,) or box.dtype.kind not in 'iu'):
        raise ValueError('Metric camera and integer crop rectangle required')
    x0,y0,x1,y1=map(int,box);h,w=shape
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Crop exceeds source image')
    out=K.copy();out[:2,2]-=[x0,y0]
    # The inference API assumes a centered principal point. Do not silently
    # apply that model to arbitrary off-center crops.
    if not np.allclose(out[:2,2],[(x1-x0)/2,(y1-y0)/2],atol=.01):
        raise ValueError('Depth crop must remain centered on the camera principal point')
    return out,float(np.rad2deg(2*np.arctan((x1-x0)/(2*out[0,0]))))


def floor_align(depth, K, plane, rectangles, excluded, *, min_pixels=500,
                max_p90_residual_m=.05, max_region_gain_ratio=1.05):
    z=np.asarray(depth,float);K=np.asarray(K,float);plane=np.asarray(plane,float);excluded=np.asarray(excluded,bool)
    if (z.ndim!=2 or excluded.shape!=z.shape or K.shape!=(3,3) or plane.shape!=(4,)
            or not np.isfinite(z).all() or np.any(z<0) or not np.isfinite(K).all()
            or min(K[0,0],K[1,1])<=0 or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1)):
        raise ValueError('Finite depth, camera, unit floor plane and exclusion mask required')
    if len(rectangles)<2:raise ValueError('At least two independently reviewed floor regions required')
    targets=[];values=[];flags=[];regions=[];h,w=z.shape
    for region,rect in enumerate(rectangles):
        x0,y0,x1,y1=rect
        if any(type(v) not in (int,np.int64,np.int32) for v in rect) or not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('Native integer floor rectangle required')
        yy,xx=np.mgrid[y0:y1,x0:x1];rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        denominator=rays@plane[:3];target=np.zeros_like(denominator,float)
        np.divide(-plane[3],denominator,out=target,where=np.abs(denominator)>1e-8)
        observed=z[y0:y1,x0:x1];valid=(observed>0)&(target>0)&~excluded[y0:y1,x0:x1]
        targets.extend(target[valid]);values.extend(observed[valid]);flags.extend(((xx+yy)%2==0)[valid]);regions.extend([region]*int(valid.sum()))
    target=np.asarray(targets);observed=np.asarray(values);fit=np.asarray(flags,bool);region=np.asarray(regions)
    reasons=[]
    if any(np.sum(fit&(region==i))<min_pixels or np.sum(~fit&(region==i))<min_pixels for i in range(len(rectangles))):
        return z.copy(),dict(accepted=False,reasons=['insufficient_independent_floor_pixels'],gain=None,proposal_only=True)
    gain=float(np.median(target[fit]/observed[fit]));gains=[float(np.median(target[fit&(region==i)]/observed[fit&(region==i)])) for i in range(len(rectangles))]
    residual=np.abs(observed[~fit]*gain-target[~fit]);p90=float(np.quantile(residual,.9))
    if not .5<=gain<=2.:reasons.append('scale_hypothesis_out_of_bounds')
    if max(gains)/min(gains)>max_region_gain_ratio:reasons.append('floor_regions_disagree')
    if p90>max_p90_residual_m:reasons.append('heldout_floor_residual')
    return z*gain,dict(accepted=not reasons,reasons=reasons,gain=gain,region_gains=gains,
        heldout_p90_residual_m=p90,fit_pixels=int(fit.sum()),heldout_pixels=int((~fit).sum()),
        proposal_only=True,object_pose_used=False,metric_truth_known=False)


def spatial_floor_holdout(depth, K, plane, rectangles, excluded, *, fit_rectangles,
                          min_pixels=500, max_p90_residual_m=.05):
    """Check fixed calibration on spatially separate, reviewed floor patches.

    No scale is fitted here. The plane remains predicted evidence, not measured
    truth; many neighboring pixels are not independent observations.
    """
    z=np.asarray(depth,float);K=np.asarray(K,float);plane=np.asarray(plane,float)
    excluded=np.asarray(excluded,bool)
    if (z.ndim!=2 or excluded.shape!=z.shape or K.shape!=(3,3) or plane.shape!=(4,)
            or not np.isfinite(z).all() or np.any(z<0) or not np.isfinite(K).all()
            or min(K[0,0],K[1,1])<=0 or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1)
            or type(min_pixels) is not int or min_pixels<1
            or not np.isfinite(max_p90_residual_m) or max_p90_residual_m<=0):
        raise ValueError('Finite depth, calibrated camera, unit plane and positive limits required')
    if not rectangles or not fit_rectangles:
        raise ValueError('Explicit spatial holdout and calibration rectangles required')
    h,w=z.shape
    def valid_rectangle(rect):
        if (len(rect)!=4 or any(type(v) not in (int,np.int32,np.int64) for v in rect)
                or not (0<=rect[0]<rect[2]<=w and 0<=rect[1]<rect[3]<=h)):
            raise ValueError('Native integer rectangle within image required')
    for rect in list(rectangles)+list(fit_rectangles):valid_rectangle(rect)
    regions=[];reasons=[]
    for i,(x0,y0,x1,y1) in enumerate(rectangles):
        if any(max(x0,a)<min(x1,c) and max(y0,b)<min(y1,d) for a,b,c,d in fit_rectangles):
            raise ValueError('Spatial holdout must not overlap calibration rectangles')
        yy,xx=np.mgrid[y0:y1,x0:x1]
        rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        denominator=rays@plane[:3];target=np.zeros_like(denominator,float)
        np.divide(-plane[3],denominator,out=target,where=np.abs(denominator)>1e-8)
        observed=z[y0:y1,x0:x1]
        valid=(target>0)&(observed>0)&~excluded[y0:y1,x0:x1]
        residual=np.abs(observed[valid]-target[valid]);count=int(valid.sum())
        p90=float(np.quantile(residual,.9)) if count else None
        if count<min_pixels:reasons.append(f'insufficient_spatial_holdout_pixels_{i}')
        if p90 is not None and p90>max_p90_residual_m:reasons.append(f'spatial_holdout_residual_{i}')
        regions.append(dict(pixels=count,p90_residual_m=p90,rectangle=[x0,y0,x1,y1]))
    return dict(accepted=not reasons,reasons=reasons,regions=regions,scale_refitted=False,
                metric_truth_known=False,pixels_are_independent_samples=False)


def audit_probe(result, *, adjacent_step_notice_m=.15, cross_scale_notice_m=.25):
    """Describe foreground inconsistencies separately from floor acceptance.

    Notice thresholds are reporting aids, not tracking acceptance limits.
    Temporal jumps are measured only between consecutive native frame IDs;
    sparse sampling cannot establish continuity inside unsampled intervals.
    No preferred depth variant or repaired surface is returned.
    """
    limits=np.asarray([adjacent_step_notice_m,cross_scale_notice_m],float)
    if not np.isfinite(limits).all() or np.any(limits<=0):
        raise ValueError('Positive finite diagnostic limits required')
    rows=result['rows'];ids=[row['source_frame'] for row in rows]
    if not ids or any(type(i) is not int or i<0 for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('Sorted unique native source frames required')
    variants={};disagreement=[]
    names=set(rows[0]['variants'])
    for row in rows:
        if set(row['variants'])!=names:
            raise ValueError('Depth variants must match across source frames')
    for name in sorted(names):
        medians=[];missing=[];floor_failed=[]
        for row in rows:
            value=row['variants'][name];q=value['object_depth_quantiles_m']
            if not value['accepted']:floor_failed.append(row['source_frame'])
            if not q:
                medians.append(None);missing.append(row['source_frame']);continue
            if len(q)!=3 or not np.isfinite(q).all() or min(q)<=0 or list(q)!=sorted(q):
                raise ValueError('Ordered positive finite object-depth quantiles required')
            medians.append(float(q[1]))
        pairs=[]
        for j in range(1,len(ids)):
            if ids[j]-ids[j-1]!=1 or medians[j] is None or medians[j-1] is None:continue
            delta=medians[j]-medians[j-1]
            pairs.append(dict(source_frames=ids[j-1:j+1],signed_median_change_m=delta,
                              exceeds_notice=bool(abs(delta)>adjacent_step_notice_m)))
        variants[name]=dict(floor_pass_count=len(rows)-len(floor_failed),floor_failed_frames=floor_failed,
            missing_foreground_frames=missing,adjacent_native_pairs=pairs,
            adjacent_jump_count=sum(p['exceeds_notice'] for p in pairs),median_depths_m=medians)
    if {'full','crop'}<=names:
        for j,frame in enumerate(ids):
            a,b=variants['full']['median_depths_m'][j],variants['crop']['median_depths_m'][j]
            if a is not None and b is not None and abs(a-b)>cross_scale_notice_m:
                disagreement.append(dict(source_frame=frame,absolute_median_difference_m=abs(a-b)))
    return dict(variants=variants,cross_scale_disagreement=disagreement,
        unsampled_intervals=[[a+1,b-1] for a,b in zip(ids,ids[1:]) if b>a+1],
        diagnostic_limits_m=dict(adjacent_step_notice=adjacent_step_notice_m,cross_scale_notice=cross_scale_notice_m),
        floor_pass_is_object_validation=False,continuous_motion_established=False,
        preferred_variant=None,reference_approved=False)


def qualify_for_pose_validation(result, expected_source_frames, *, variant='crop',
                                min_visible_pixels=100, min_visible_coverage=.95,
                                max_adjacent_depth_change_m=.15):
    """Gate a depth *experiment* before expensive pose validation.

    Source IDs must match the caller's declared samples. A floor fit can pass
    while foreground depth is missing or jumps, so check those independently.
    Passing only permits native pose tests; sparse samples do not establish
    continuity, absolute depth accuracy or an accepted motion reference.
    """
    expected=list(expected_source_frames)
    if (not expected or any(type(i) is not int or i<0 for i in expected)
            or expected!=sorted(set(expected))):
        raise ValueError('Sorted unique expected native source frames required')
    if [r['source_frame'] for r in result['rows']]!=expected:
        raise ValueError('Probe does not cover the declared native samples')
    if (type(min_visible_pixels) is not int or min_visible_pixels<1
            or not np.isfinite(min_visible_coverage) or not 0<min_visible_coverage<=1
            or not np.isfinite(max_adjacent_depth_change_m) or max_adjacent_depth_change_m<=0):
        raise ValueError('Positive finite foreground qualification limits required')
    audit=audit_probe(result,adjacent_step_notice_m=max_adjacent_depth_change_m)
    if variant not in audit['variants']:raise ValueError('Requested depth variant is absent')
    reasons=[]
    for row in result['rows']:
        v=row['variants'][variant];count=v['object_pixels'];valid=v['valid_object_pixels']
        if (type(count) is not int or type(valid) is not int or count<0 or not 0<=valid<=count
                or type(v['accepted']) is not bool):
            raise ValueError('Explicit floor decision and consistent native foreground counts required')
        if not v['accepted']:reasons.append(f"floor_{row['source_frame']}")
        if count<min_visible_pixels or valid/max(1,count)<min_visible_coverage:
            reasons.append(f"coverage_{row['source_frame']}")
        if not v['object_depth_quantiles_m']:reasons.append(f"missing_depth_{row['source_frame']}")
    if audit['variants'][variant]['adjacent_jump_count']:reasons.append('adjacent_depth_jump')
    return dict(passed=not reasons,reasons=reasons,audit=audit,depth_variant=variant,
        expected_source_frames=expected,reference_approved=False,
        requires_native_gpu_pose_checks=True,metric_truth_known=False,
        note='Sampled RGB predictions calibrated on held-out floor pixels; native pose and full-sequence qualification remain required.')


def probe(request):
    import h5py,torch
    from PIL import Image
    from moge.model.v2 import MoGeModel
    from real2sim2real.artifacts import digest,write_json
    request=Path(request);spec=json.loads(request.read_text());out=Path(spec['output'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise FileExistsError('Preserve previous depth probe')
    source=Path(spec['inputs']);bundle=np.load(source/'evidence.npz',allow_pickle=False)
    ids=bundle['source_frame_ids'];K=bundle['K'];box=spec['crop_xyxy'];x0,y0,x1,y1=box
    objects=bundle['object_masks'];humans=bundle['human_masks'];shape=objects.shape[1:]
    cropK,cropfov=crop_camera(K,box,shape);_,fullfov=crop_camera(K,[0,0,shape[1],shape[0]],shape)
    model=MoGeModel.from_pretrained(spec['checkpoint']).to('cuda').eval()
    rows=[]
    with h5py.File(source/'rgb.h5') as rgb:
        for local,frame in enumerate(ids):
            image=np.array(Image.open(io.BytesIO(rgb['frames'][local].tobytes())).convert('RGB'))
            row=dict(source_frame=int(frame),object_pixels=int((objects[local]&~humans[local]).sum()),variants={})
            for name,picture,camera,fov,offset in [('full',image,K,fullfov,(0,0)),('crop',image[y0:y1,x0:x1],cropK,cropfov,(x0,y0))]:
                tensor=torch.from_numpy(np.ascontiguousarray(picture)).permute(2,0,1).to('cuda').float()/255
                with torch.inference_mode():prediction=model.infer(tensor,resolution_level=9,force_projection=True,apply_mask=True,use_fp16=True,fov_x=fov)
                depth=prediction['depth'].detach().cpu().numpy().astype(np.float32)
                valid=prediction['mask'].detach().cpu().numpy().astype(bool)&np.isfinite(depth)&(depth>0)
                depth[~valid]=0
                ox,oy=offset;hh,ww=depth.shape
                obj=objects[local,oy:oy+hh,ox:ox+ww];human=humans[local,oy:oy+hh,ox:ox+ww]
                rectangles=[[a-ox,b-oy,c-ox,d-oy] for a,b,c,d in spec['floor_rectangles_xyxy']]
                aligned,receipt=floor_align(depth,camera,bundle['plane_camera'],rectangles,obj|human)
                if spec.get('floor_holdout_rectangles_xyxy'):
                    heldout=[[a-ox,b-oy,c-ox,d-oy] for a,b,c,d in spec['floor_holdout_rectangles_xyxy']]
                    check=spatial_floor_holdout(aligned,camera,bundle['plane_camera'],heldout,
                                                obj|human,fit_rectangles=rectangles)
                    receipt['spatial_holdout']=check
                    receipt['accepted']=receipt['accepted'] and check['accepted']
                    receipt['reasons']+=check['reasons']
                visible=obj&~human&valid
                receipt.update(object_depth_quantiles_m=np.quantile(aligned[visible],[.1,.5,.9]).tolist() if visible.any() else [],
                    valid_object_pixels=int(visible.sum()),object_pixels=int((obj&~human).sum()),
                    object_observed=bool((obj&~human).any()),camera_fov_x_deg=fov)
                np.savez_compressed(out/f'{int(frame):06d}_{name}.npz',raw_depth_m=depth,aligned_depth_proposal_m=aligned.astype(np.float32),valid=valid,K=camera)
                row['variants'][name]=receipt
                del prediction,tensor
            rows.append(row);write_json(out/'progress.json',dict(completed=len(rows),total=len(ids),last=row,proposal_only=True))
            print('DEPTH_PROBE',int(frame),row['variants'],flush=True)
    result=dict(schema='real2sim2real.foreground-depth-probe/1',rows=rows,proposal_only=True,
        full_reference_approved=False,hidden_motion_observed=False,source_request_sha256=digest(request),
        note='Same-model cross-scale predictions calibrated on predicted floor; not independent ground truth. No source depth was overwritten.')
    write_json(out/'RESULTS.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);probe(p.parse_args().request)
