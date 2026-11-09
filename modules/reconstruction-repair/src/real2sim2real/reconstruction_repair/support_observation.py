"""Unaccepted rigid-pose seeds from an observed support contact pixel.

The contact pixel and stationary plane must be reviewed observations, not a
grasp label or an inferred success signal. This module does not infer contact,
modify depth or mesh, or approve the resulting pose for tracking/physics.
"""
import numpy as np
from real2sim2real.reconstruction.recovery_quality import rigid


def support_pixel_seed(rotation,contact_local,pixel_xy,intrinsics,plane_camera,*,resting_evidence):
    r,p,uv,K,plane=map(lambda v:np.asarray(v,float),(rotation,contact_local,pixel_xy,intrinsics,plane_camera))
    pose=np.eye(4)
    if r.shape!=(3,3):raise ValueError('Rigid rotation required')
    pose[:3,:3]=r
    if (not rigid(pose) or p.shape!=(3,) or uv.shape!=(2,) or K.shape!=(3,3) or plane.shape!=(4,)
            or not all(np.isfinite(v).all() for v in (p,uv,K,plane))
            or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1])
            or not np.isclose(np.linalg.norm(plane[:3]),1,atol=1e-5)
            or not isinstance(resting_evidence,str) or not resting_evidence.strip()):
        raise ValueError('Metric contact, calibrated pixel, unit support plane and explicit evidence required')
    ray=np.linalg.solve(K,np.r_[uv,1.]);denominator=float(plane[:3]@ray)
    if abs(denominator)<1e-8:raise ValueError('Observed ray is parallel to the support plane')
    z=-float(plane[3])/denominator
    if z<=.01:raise ValueError('Support intersection is behind/too close to the camera')
    contact=ray*z;pose[:3,3]=contact-r@p
    return pose,dict(schema='real2sim2real.support-pixel-seed/1',pixel_xy=uv.tolist(),
        contact_local_m=p.tolist(),contact_camera_m=contact.tolist(),plane_camera=plane.tolist(),
        ray_plane_denominator=denominator,depth_sensitivity_to_plane_offset=abs(1./denominator),
        sensitivity_note='Metres of optical-axis depth change per metre of plane-offset error, holding normal/pixel fixed; not an uncertainty estimate.',
        resting_evidence=resting_evidence,depth_changed=False,geometry_changed=False,
        proposal_only=True,contact_assumption_inferred=False,requires_image_depth_motion_validation=True)
