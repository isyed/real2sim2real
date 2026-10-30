"""Input-only foreground-depth/visibility preflight before tracking or repair.

An internally coherent depth patch may still be background. Temporal warnings
are diagnostics, never corrected depth or proof that real motion is impossible.
Missing observations require explicit reacquisition/inference, not silent carry.
"""
from dataclasses import asdict,dataclass
import numpy as np
from scipy.ndimage import binary_erosion


@dataclass(frozen=True)
class ForegroundConfig:
    erosion_pixels:int=2
    min_depth_pixels:int=100
    max_relative_depth_iqr:float=.35
    max_surface_depth_rate_m_s:float=2.4
    surface_sampling_slack_m:float=.02

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('Positive finite evidence bounds required')
        for value in [self.erosion_pixels,self.min_depth_pixels]:
            if int(value)!=value:raise ValueError('Integer pixel bounds required')


def measure(depth_m,object_mask,human_mask,*,frame_id,config=ForegroundConfig()):
    config.validate();z=np.asarray(depth_m);obj=np.asarray(object_mask);human=np.asarray(human_mask)
    if (type(frame_id) is not int or frame_id<0 or z.ndim!=2 or z.dtype.kind!='f'
        or not np.isfinite(z).all() or np.any(z<0) or obj.shape!=z.shape or human.shape!=z.shape
        or obj.dtype!=bool or human.dtype!=bool):raise ValueError('Native finite metre depth, boolean masks and frame ID required')
    visible=obj&~human;valid=binary_erosion(visible,iterations=config.erosion_pixels)&(z>0)&(z<65.535)
    count=int(valid.sum());median=None;spread=None
    if count:
        q=np.percentile(z[valid],[25,50,75]);median=float(q[1]);spread=float((q[2]-q[0])/q[1])
    reasons=[]
    if not obj.any():reasons.append('object_mask_missing')
    if count<config.min_depth_pixels:reasons.append('insufficient_visible_depth')
    if spread is not None and spread>config.max_relative_depth_iqr:reasons.append('incoherent_foreground_depth')
    return dict(frame_id=frame_id,object_pixels=int(obj.sum()),visible_pixels=int(visible.sum()),
        depth_pixels=count,median_depth_m=median,relative_depth_iqr=spread,
        observation_available=not reasons,reasons=reasons,metric_accuracy_verified=False)


def audit(measurements,timestamps_s,*,config=ForegroundConfig()):
    config.validate();rows=[dict(row) for row in measurements];times=np.asarray(timestamps_s,float)
    if (not rows or [r['frame_id'] for r in rows]!=list(range(len(rows))) or times.shape!=(len(rows),)
        or not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError('Complete consecutive source frames and increasing timestamps required')
    for r in rows:
        median=r['median_depth_m'];spread=r['relative_depth_iqr']
        if (type(r['observation_available']) is not bool or type(r['depth_pixels']) is not int or r['depth_pixels']<0
            or (median is not None and (not np.isfinite(median) or median<=0))
            or (spread is not None and (not np.isfinite(spread) or spread<0))
            or (r['observation_available'] and (median is None or spread is None or r['depth_pixels']<config.min_depth_pixels
                or spread>config.max_relative_depth_iqr))):raise ValueError('Inconsistent foreground evidence row')
    warnings=[];previous=None
    for i,row in enumerate(rows):
        row['temporal_depth_warning']=False
        if row['observation_available']:
            if previous is not None and i==previous+1:
                delta=abs(row['median_depth_m']-rows[previous]['median_depth_m'])
                limit=config.max_surface_depth_rate_m_s*(times[i]-times[previous])+config.surface_sampling_slack_m
                row.update(surface_depth_change_m=float(delta),surface_depth_warning_bound_m=float(limit))
                if delta>limit:
                    row['temporal_depth_warning']=True;warnings.append(i)
            previous=i
    gaps=[];i=0
    while i<len(rows):
        if rows[i]['observation_available']:i+=1;continue
        start=i
        while i<len(rows) and not rows[i]['observation_available']:i+=1
        gaps.append(dict(first_frame=start,last_frame=i-1,frames=i-start,
            duration_s=float(times[i-1]-times[start]),
            reasons=sorted({reason for row in rows[start:i] for reason in row['reasons']}),
            last_observed_frame=start-1 if start else None,reacquisition_frame=i if i<len(rows) else None,
            hidden_object_motion_inferred=False))
    return dict(schema='real2sim2real.foreground-visibility-audit/1',config=asdict(config),rows=rows,
        source_frames=len(rows),unavailable_observation_spans=gaps,temporal_depth_warning_frames=warnings,
        full_observation_workflow_ready=not gaps and not warnings,
        observed_frames=sum(row['observation_available'] for row in rows),
        fresh_registration_needed_after_gaps=[gap['reacquisition_frame'] for gap in gaps if gap['reacquisition_frame'] is not None],
        depth_modified=False,poses_modified=False,hidden_frames_filled=False,reference_approved=False,
        limitations=['Missing masks can reflect occlusion or segmentation failure; inspect source RGB.',
                    'Temporal surface-depth changes can reflect changing visible surfaces; warnings are not true-motion measurements.',
                    'Available depth counts and narrow spread do not establish correct foreground depth.',
                    'This audit does not reconstruct hidden motion or approve a replacement reference.'])


def overlay(rgb,object_mask,row):
    """Visualize source evidence only; no mask is drawn where none was observed."""
    a=np.asarray(rgb);m=np.asarray(object_mask)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8 or m.shape!=a.shape[:2] or m.dtype!=bool:
        raise ValueError('RGB uint8 and a same-grid boolean object mask required')
    out=a.copy();color=([245,75,65] if row.get('temporal_depth_warning') or not row['observation_available'] else [40,220,150])
    out[m]=(out[m]*.45+np.asarray(color)*.55).astype(np.uint8)
    return out


def artifact_audit(source,output,*,camera='front_stereo_camera_left',config=ForegroundConfig()):
    """Stream a portable artifact's complete native timeline on CPU.

    Produces an audit receipt, never a reconstruction replacement. Heavy file
    readers are optional imports; no predictor or controller is loaded.
    """
    from pathlib import Path
    import io
    import h5py
    from PIL import Image
    from real2sim2real.artifacts import load,digest,write_json
    source,output=Path(source),Path(output);config.validate()
    if output.exists():raise FileExistsError('Preserve previous visibility evidence')
    artifact=load(source,kinds={'reconstruction'})
    required=['geometry','depth','object_masks','human_masks']
    if not all(role in artifact['files'] for role in required):raise ValueError('Geometry, depth and both foreground masks required')
    files={role:source/artifact['files'][role]['path'] for role in required}
    with np.load(files['geometry'],allow_pickle=False) as g:
        ids=g['frame_ids'];times=g['timestamps_s']
        if not np.array_equal(ids,np.arange(len(ids))):raise ValueError('Complete zero-based source timeline required')
    rows=[]
    with h5py.File(files['depth']) as depth,h5py.File(files['object_masks']) as obj,h5py.File(files['human_masks']) as human:
        encoded=depth[f'aligned/{camera}']
        if len(encoded)!=len(ids) or obj['frames'].shape[0]!=len(ids) or human['frames'].shape!=obj['frames'].shape:
            raise ValueError('Native depth/mask/source coverage differs')
        for i in range(len(ids)):
            z=np.asarray(Image.open(io.BytesIO(encoded[i].tobytes())),np.float32)/1000
            rows.append(measure(z,obj['frames'][i]!=0,human['frames'][i]!=0,frame_id=i,config=config))
    result=audit(rows,times,config=config)
    result['source_identity']=dict(artifact_sha256=digest(source/'artifact.json'),
        files={role:artifact['files'][role]['sha256'] for role in required},
        own_predicted_inputs_only=True,camera=camera)
    output.mkdir(parents=True);write_json(output/'report.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--camera',default='front_stereo_camera_left');a=p.parse_args()
    result=artifact_audit(a.input,a.output,camera=a.camera)
    print(dict(source_frames=result['source_frames'],full_observation_workflow_ready=result['full_observation_workflow_ready'],
               unavailable_spans=len(result['unavailable_observation_spans']),temporal_warnings=len(result['temporal_depth_warning_frames'])))
