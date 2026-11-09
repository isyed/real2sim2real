"""Explicit resting-surface hypotheses that preserve the source projection.

Only vertices below a caller-declared support are moved. Camera-ray projection
preserves the initial outline; opt-in normal projection minimizes displacement
but can change that outline. This is a geometry proposal, not metric truth. It is
useful for testing hidden lower-surface errors without globally resizing an
object whose visible foreground already agrees with depth.
"""
import numpy as np
from real2sim2real.reconstruction.recovery_quality import rigid


def project_below_support(vertices, faces, pose_camera, plane_camera, *,
                          resting_evidence, clearance_m=.001,
                          max_displacement_m=.10, max_changed_fraction=.15,
                          max_relative_volume_change=.15,
                          projection_mode='camera_ray'):
    v,p,plane=map(lambda x:np.asarray(x,dtype=float),(vertices,pose_camera,plane_camera))
    f=np.asarray(faces)
    limits=np.asarray([clearance_m,max_displacement_m,max_changed_fraction,max_relative_volume_change],float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1]!=3 or len(f)<4 or f.dtype.kind not in 'iu'
            or f.min()<0 or f.max()>=len(v) or not rigid(p)
            or plane.shape!=(4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6) or plane[3]<=0
            or not isinstance(resting_evidence,str) or not resting_evidence.strip()
            or not np.isfinite(limits).all() or clearance_m<0 or clearance_m>=plane[3]
            or projection_mode not in ('camera_ray','support_normal')
            or max_displacement_m<=0 or not 0<max_changed_fraction<=1
            or not 0<max_relative_volume_change<1):
        raise ValueError('Metric mesh, camera pose, unit oriented support, explicit resting evidence and finite bounds required')
    camera=v@p[:3,:3].T+p[:3,3]
    if camera[:,2].min()<=.01:raise ValueError('Mesh crosses camera near plane')
    signed=camera@plane[:3]+plane[3];changed=signed<clearance_m-1e-10
    factors=np.ones(len(v))
    if projection_mode=='camera_ray':
        factors[changed]=-(plane[3]-clearance_m)/(camera[changed]@plane[:3])
        if np.any(factors<=0):raise ValueError('Support intersection crosses camera origin')
        corrected=camera*factors[:,None]
    else:
        corrected=camera.copy()
        corrected[changed]+=(clearance_m-signed[changed,None])*plane[:3]
        if corrected[:,2].min()<=.01:raise ValueError('Support projection crosses camera near plane')
    # Invert explicitly: a supplied predicted rotation may carry admissible
    # float32 roundoff. Preserve all untouched local vertices bit-for-bit.
    proposed=np.linalg.solve(p[:3,:3],(corrected-p[:3,3]).T).T
    proposed[~changed]=v[~changed]
    def geometry(points):
        tri=points[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        return dict(signed_volume_m3=float(np.einsum('ij,ij->',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6),
                    near_zero_area_faces=int((np.linalg.norm(normal,axis=1)<2e-14).sum()))
    before,after=geometry(v),geometry(proposed)
    volume_change=abs(after['signed_volume_m3']/before['signed_volume_m3']-1) if abs(before['signed_volume_m3'])>1e-12 else None
    distance=np.linalg.norm(corrected-camera,axis=1);fraction=float(changed.mean());reasons=[]
    if distance.max()>max_displacement_m:reasons.append('displacement_bound')
    if fraction>max_changed_fraction:reasons.append('changed_fraction_bound')
    if volume_change is None or volume_change>max_relative_volume_change:reasons.append('volume_change_bound')
    if before['signed_volume_m3']*after['signed_volume_m3']<=0:reasons.append('signed_volume_invalid')
    if after['near_zero_area_faces']>before['near_zero_area_faces']:reasons.append('triangle_degeneracy_increased')
    actual_camera=proposed@p[:3,:3].T+p[:3,3]
    uv_before=camera[:,:2]/camera[:,2:];uv_after=actual_camera[:,:2]/actual_camera[:,2:]
    return proposed,dict(schema='real2sim2real.support-surface-proposal/1',numerical_bounds_passed=not reasons,reasons=reasons,
        resting_evidence=resting_evidence,clearance_m=clearance_m,changed_vertices=int(changed.sum()),
        total_vertices=len(v),changed_fraction=fraction,maximum_displacement_m=float(distance.max()),
        minimum_ray_factor=float(factors.min()) if projection_mode=='camera_ray' else None,relative_signed_volume_change=volume_change,
        geometry_before=before,geometry_after=after,
        minimum_support_height_before_m=float(signed.min()),
        minimum_support_height_after_m=float((corrected@plane[:3]+plane[3]).min()),
        max_normalized_projection_error=float(np.abs(uv_after-uv_before).max()),
        limits=dict(max_displacement_m=max_displacement_m,max_changed_fraction=max_changed_fraction,
                    max_relative_volume_change=max_relative_volume_change),
        vertices_only_below_reviewed_support=True,pose_unchanged=True,topology_unchanged=True,
        projection_mode=projection_mode,initial_projection_preserved=projection_mode=='camera_ray',
        geometry_changed=bool(changed.any()),absolute_metric_accuracy_established=False,
        source_depth_validated=False,heldout_image_validated=False,physics_validated=False,
        complete_reference_approved=False,actor_training_approved=False,
        limitations=['A predicted plane and resting phase require independent source evidence.',
                    'Unchanged initial projection does not prove hidden geometry is correct.',
                    'Recompute inertia and validate held-out images, depth, and CPU/GPU contacts.'])
