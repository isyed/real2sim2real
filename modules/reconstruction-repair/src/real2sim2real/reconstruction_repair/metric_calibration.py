"""Coarse size diagnosis from RGB masks and independently predicted metric depth.

This produces a mesh hypothesis, NEVER a repaired trajectory. Resizing a mesh
does not repair old pose/depth estimates: fresh tracking and held-out validation
are mandatory. Projected extent underestimates foreshortened objects; the upper
fit quantile assumes several views expose nearly the full object diameter.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from .bundle import arrays_atomic, arrays_read, bundle_read, file_sha256, json_atomic
from .geometry import object_camera_centers


@dataclass(frozen=True)
class CalibrationConfig:
    max_scale_ratio: float = 5.
    upper_fit_quantile: float = .9
    max_relative_scale_disagreement: float = .20
    max_relative_depth_iqr: float = .35
    max_floor_height_drift: float = .10
    max_floor_normal_angle_deg: float = 5.
    mask_erosion_pixels: int = 2
    min_depth_pixels: int = 100
    min_split_frames: int = 3

    def validate(self):
        for key,value in asdict(self).items():
            if not np.isfinite(value) or value<=0:raise ValueError(f'Invalid calibration bound: {key}')
        if self.max_scale_ratio<1 or not .5<=self.upper_fit_quantile<=1:
            raise ValueError('Invalid scale ratio or quantile')
        if any(getattr(self,k)!=int(getattr(self,k)) for k in ('mask_erosion_pixels','min_depth_pixels','min_split_frames')):
            raise ValueError('Pixel and frame counts must be integer')
        if self.min_split_frames<3:
            raise ValueError('At least three frames per split are required')


def diameter(points):
    """Exact convex-hull diameter with bounded pairwise working memory."""
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1] not in (2,3) or len(p)<3 or not np.isfinite(p).all():
        raise ValueError('Need finite 2D/3D points')
    # Project to the actual affine span: planar meshes are legitimate.
    center=p.mean(axis=0);_,s,v=np.linalg.svd(p-center,full_matrices=False)
    rank=int(np.count_nonzero(s>max(s[0]*1e-10,1e-12)))
    if rank<1:raise ValueError('Zero extent')
    q=(p-center)@v[:rank].T
    if rank==1:return float(np.ptp(q[:,0]))
    try:vertices=q[ConvexHull(q).vertices]
    except QhullError as exc:raise ValueError('Degenerate extent evidence') from exc
    return max(float(cdist(vertices[start:start+256],vertices).max()) for start in range(0,len(vertices),256))


def diagnose(bundle, evidence, provenance, config=CalibrationConfig()):
    config.validate();g,o=bundle.geometry,bundle.observations
    required={'frame_ids','depth_m','fit','floor_planes','floor_valid'}
    if set(evidence)!=required:raise ValueError('Metric evidence fields differ from the explicit contract')
    if (provenance.get('schema')!='r2s2r_engines.metric-calibration-evidence/1'
        or not provenance.get('source') or not provenance.get('floor_roi_source')
        or provenance.get('clean_trajectory_used') is not False
        or provenance.get('input_manifest_sha256')!=file_sha256(bundle.root/'manifest.json')):
        raise ValueError('Explicit source, floor review, input hash and RGB-only provenance are required')
    if 'object_mask' not in o:raise ValueError('Object masks are required')
    ids=np.asarray(evidence['frame_ids']);fit=np.asarray(evidence['fit'])
    n=len(ids);depth=np.asarray(evidence['depth_m']);planes=np.asarray(evidence['floor_planes'])
    valid_floor=np.asarray(evidence['floor_valid'])
    if (ids.dtype.kind not in 'iu' or ids.shape!=(n,) or np.any(np.diff(ids)<=0)
        or not np.isin(ids,g['frame_ids']).all() or fit.shape!=(n,) or not np.isin(fit,[0,1]).all()
        or planes.shape!=(n,4) or valid_floor.shape!=(n,) or not np.isin(valid_floor,[0,1]).all()
        or not np.isfinite(planes).all() or not np.allclose(np.linalg.norm(planes[:,:3],axis=1),1,atol=1e-5)
        or np.any(planes[:,3]<=0)):
        raise ValueError('Invalid frame split or oriented metric floor planes')
    fit=fit.astype(bool)
    if min(int(fit.sum()),int((~fit).sum()))<config.min_split_frames:
        raise ValueError('Need at least three predeclared fit and three held-out frames')
    if depth.shape!=(n,*o['object_mask'].shape[1:]) or not np.isfinite(depth).all():
        raise ValueError('Depth and mask grids differ or depth is nonfinite')
    original_diameter=diameter(g['vertices'][np.unique(g['faces'])])*float(g['object_scale'])
    centers=object_camera_centers(g);rows=[];factors=[];reasons=[]
    for j,frame in enumerate(ids):
        i=int(np.searchsorted(g['frame_ids'],frame));mask=o['object_mask'][i].astype(bool).copy()
        if 'occluder_mask' in o:mask &= ~o['occluder_mask'][i].astype(bool)
        inner=binary_erosion(mask,iterations=int(config.mask_erosion_pixels))
        valid=inner & (depth[j]>0) & (depth[j]<65.535)
        if not bool(o['reliable'][i]) or int(valid.sum())<config.min_depth_pixels:
            raise ValueError(f'Insufficient visible, reliable depth at frame {frame}')
        z=np.percentile(depth[j][valid],[25,50,75]);y,x=np.where(mask)
        rays=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(g['intrinsics'][i]).T
        angular_diameter=diameter(rays[:,:2]/rays[:,2:])
        extent=angular_diameter*float(z[1]);factor=extent/original_diameter
        depth_spread=float((z[2]-z[0])/z[1])
        if depth_spread>config.max_relative_depth_iqr:reasons.append(f'depth_incoherent:{int(frame)}')
        rows.append(dict(frame_id=int(frame),split='fit' if fit[j] else 'held_out',depth_median_m=float(z[1]),
            depth_relative_iqr=depth_spread,visible_depth_pixels=int(valid.sum()),
            projected_metric_diameter_m=extent,scale_observation=factor,
            old_object_center_depth_m=float(centers[i,2]),
            old_center_minus_observed_surface_depth_m=float(centers[i,2]-z[1])))
        factors.append(factor)
    factors=np.asarray(factors)
    # Only fit frames determine the hypothesis. Held-out rows can only veto it.
    scale=float(np.quantile(factors[fit],config.upper_fit_quantile))
    if not 1/config.max_scale_ratio<=scale<=config.max_scale_ratio:reasons.append('scale_outside_declared_bound')
    fit_disagreement=float(np.max(np.abs(factors[fit]/scale-1)))
    held_disagreement=float(np.max(np.abs(factors[~fit]/scale-1)))
    disagreement=np.abs(factors/scale-1)
    for row,value in zip(rows,disagreement):
        row['relative_scale_disagreement']=float(value)
        row['scale_consistency_passed']=bool(value<=config.max_relative_scale_disagreement)
    if fit_disagreement>config.max_relative_scale_disagreement:reasons.append('fit_extent_disagreement')
    if held_disagreement>config.max_relative_scale_disagreement:reasons.append('heldout_extent_disagreement')
    floor_height=float(np.median(planes[fit,3]));floor_drift=float(np.max(np.abs(planes[:,3]/floor_height-1)))
    normal=np.mean(planes[fit,:3],axis=0);normal/=np.linalg.norm(normal)
    angle=float(np.rad2deg(np.arccos(np.clip(planes[:,:3]@normal,-1,1))).max())
    if not valid_floor.all():reasons.append('floor_fit_failed')
    if floor_drift>config.max_floor_height_drift:reasons.append('metric_depth_floor_drift')
    if angle>config.max_floor_normal_angle_deg:reasons.append('floor_orientation_inconsistent')
    return dict(schema='r2s2r_engines.object-metric-calibration/1',status='candidate_requires_retracking' if not reasons else 'rejected',
        candidate_ready=not reasons,scale_multiplier=scale,original_mesh_diameter_m=original_diameter,
        candidate_mesh_diameter_m=original_diameter*scale,fit_relative_disagreement=fit_disagreement,
        heldout_relative_disagreement=held_disagreement,floor_relative_height_drift=floor_drift,
        floor_max_normal_angle_deg=angle,reasons=sorted(set(reasons)),measurements=rows,config=asdict(config),
        inconsistent_extent_frame_ids=ids[disagreement>config.max_relative_scale_disagreement].tolist(),
        old_poses_valid_for_resized_mesh=False,requires_retracking=True,approved_for_use=False,
        human_changed=False,limitations=[
            'Predicted human-aligned depth is not ground truth; stable floor does not establish absolute metric scale.',
            'Projected visible extent assumes multiple views expose nearly the full diameter; foreshortening/occlusion bias remains.',
            'Surface median depth is not the object center; their difference is a diagnostic, not a translation correction.',
            'No pose or human edits are generated. Fresh tracking, held-out image/depth/support review and physics are still required.'])


def calibrate_object(source: Path, evidence_path: Path, provenance_path: Path, output: Path, config=CalibrationConfig()):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Use a new calibration output; existing experiments are immutable')
    bundle=bundle_read(source);provenance=json.loads(Path(provenance_path).read_text())
    evidence=arrays_read(Path(evidence_path))
    if provenance.get('evidence_sha256')!=file_sha256(Path(evidence_path)):
        raise ValueError('Metric evidence hash differs')
    report=diagnose(bundle,evidence,provenance,config)
    report['provenance']=provenance
    report['source_sha256']={p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    output.mkdir(parents=True)
    # A calibration is an independently retained stage, not a transient score
    # attached to a later tracking/smoothing decision. Keep numeric evidence so
    # the default fine-repair preflight can recompute, rather than trust a flag.
    shutil.copyfile(evidence_path, output/'evidence.npz')
    shutil.copyfile(provenance_path, output/'evidence_provenance.json')
    report['evidence_sha256'] = file_sha256(output/'evidence.npz')
    report['evidence_provenance_sha256'] = file_sha256(output/'evidence_provenance.json')
    if report['candidate_ready']:
        mesh={'vertices':bundle.geometry['vertices']*bundle.geometry['object_scale']*report['scale_multiplier'],
              'faces':bundle.geometry['faces']}
        arrays_atomic(output/'calibrated_mesh.npz',mesh)
        report['mesh_sha256']=file_sha256(output/'calibrated_mesh.npz')
    from .gate_report import write_explanation
    report['gate_explanation'] = write_explanation(report, output)
    json_atomic(output/'calibration.json',report)
    (output/'REVIEW.md').write_text('# Metric calibration\n\n'+report['status']+'\n\n'
        +f"Scale hypothesis: {report['scale_multiplier']:.4f}×. Human unchanged.\n\n"
        +('**Resize hypothesis only:** fresh tracking and validation are required before selecting it.\n\n'
          if report['candidate_ready'] else
          '**Calibration rejected:** retain the input mesh. Resolve the conflicting image/depth evidence; this result does not justify resizing.\n\n')
        +'\n'.join('- '+s for s in report['limitations'])+'\n')
    return report
