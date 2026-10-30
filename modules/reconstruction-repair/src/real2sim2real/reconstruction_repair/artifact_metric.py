"""Bounded-memory size/depth preflight for a portable reconstruction artifact.

Twelve declared source frames are sampled before looking at new tracking. The
returned bundle is calibration evidence only; it is not a full repaired clip.
"""
from pathlib import Path
import io
import numpy as np
from real2sim2real.artifacts import load,role_path,digest,write_json
from .bundle import BundleData,bundle_create,file_sha256
from .contracts import COORDINATES
from .metric_floor import floor_points,fit_floor_plane,PlaneFitError
from .metric_calibration import calibrate_object
from .metric_preflight import inspect_metric_calibration


def prepare(source, output, floor_selection, *, samples=12):
    import h5py
    from PIL import Image
    source,output=Path(source),Path(output)
    artifact=load(source,kinds={'reconstruction'})
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    with np.load(role_path(source,'geometry')) as z:g=dict(z)
    n=len(g['frame_ids']);ids=np.linspace(0,n-1,samples).round().astype(int)
    if len(np.unique(ids))!=samples or samples<6:raise ValueError('Distinct calibration frames required')
    masks=[];humans=[];depths=[];planes=[];valid=[];floor_rows=[]
    with h5py.File(role_path(source,'depth')) as d,h5py.File(role_path(source,'object_masks')) as o,h5py.File(role_path(source,'human_masks')) as h:
        if o['frames'].shape[0]!=n or h['frames'].shape!=o['frames'].shape:raise ValueError('Mask/source timeline mismatch')
        for i in ids:
            depth=np.asarray(Image.open(io.BytesIO(d['aligned/front_stereo_camera_left'][i].tobytes())),np.float32)/1000
            obj=o['frames'][i]!=0;person=h['frames'][i]!=0
            points,regions,roi=floor_points(depth,g['intrinsics'][i],person,obj,floor_selection)
            try:plane,diag=fit_floor_plane(points,regions);passed=True
            except PlaneFitError as e:plane=np.array([0.,-1.,0.,1.]);diag=e.diagnostics;passed=False
            world=g['world_to_camera'][i].T@plane;world/=np.linalg.norm(world[:3])
            masks.append(obj);humans.append(person);depths.append(depth);planes.append(world);valid.append(passed)
            floor_rows.append(dict(frame=int(g['frame_ids'][i]),**diag))
    sampled={k:v[ids] if k in ['frame_ids','timestamps_s','object_rotation','object_translation','intrinsics','world_to_camera'] else v for k,v in g.items()}
    observed=dict(object_mask=np.asarray(masks),occluder_mask=np.asarray(humans),
        reliable=np.asarray([(a&~b).sum()>=100 for a,b in zip(masks,humans)]),locked=np.zeros(samples,bool))
    bundle=bundle_create(output/'bundle',BundleData(dict(sequence_id=artifact['metadata']['sequence_id'],
        world_frame_id=artifact['metadata']['world_frame_id'],coordinates=COORDINATES,
        scope='Predeclared source-frame size calibration only; not a repaired trajectory'),sampled,observed))
    evidence=output/'metric_evidence.npz'
    np.savez_compressed(evidence,frame_ids=g['frame_ids'][ids],depth_m=np.asarray(depths),fit=np.arange(samples)%2==0,
        floor_planes=np.asarray(planes),floor_valid=np.asarray(valid))
    provenance=dict(schema='r2s2r_engines.metric-calibration-evidence/1',source='Own predicted metric depth on predeclared native RGB frames',
        floor_roi_source=floor_selection,clean_trajectory_used=False,input_manifest_sha256=file_sha256(bundle.root/'manifest.json'),evidence_sha256=digest(evidence))
    proof=output/'provenance.json';write_json(proof,provenance)
    write_json(output/'floor_report.json',dict(samples=floor_rows,passed=all(valid),original_image_dimensions=floor_selection['image_size']))
    try:
        report=calibrate_object(bundle.root,evidence,proof,output/'calibration')
        status=inspect_metric_calibration(bundle,output/'calibration')
    except ValueError as error:
        # Inadequate occlusion/depth evidence is a bounded outcome, not a pass.
        report=dict(status='unavailable',candidate_ready=False,scale_multiplier=None,reasons=[str(error)])
        status=dict(passed=False,current_mesh_size_verified=False,status='calibration_unavailable',error=str(error))
        write_json(output/'calibration_error.json',report)
    result=dict(calibration=report,preflight=status,frames_sampled=g['frame_ids'][ids].tolist(),
        source_artifact_sha256=digest(source/'artifact.json'),floor_passed=all(valid),first_frame_floor_passed=bool(valid[0]),
        first_frame_plane_camera=np.linalg.inv(g['world_to_camera'][0]).T@np.asarray(planes[0]))
    result['first_frame_plane_camera']=result['first_frame_plane_camera'].tolist()
    write_json(output/'RESULTS.json',result)
    return result
