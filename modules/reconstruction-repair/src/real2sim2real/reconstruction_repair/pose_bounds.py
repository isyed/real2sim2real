"""Project rotation corrections into an explicit geodesic proposal bound.

This returns an unaccepted proposal. Recompute image, depth, support, motion and
interaction checks afterward; projection can invalidate every one of them.
It does not interpolate across unavailable observations or change translations.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def project_rotation_bound(original, proposal, *, maximum_degrees):
    a,b=np.asarray(original),np.asarray(proposal)
    if (isinstance(maximum_degrees,bool) or not np.isfinite(maximum_degrees)
            or not 0<maximum_degrees<=180):
        raise ValueError('Require a positive finite geodesic bound at most180 degrees')
    if (a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(4,4) or not len(a)
            or a.dtype.kind!='f' or b.dtype.kind!='f'
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Matching nonempty finite floating-point rigid-pose sequences required')
    for poses in (a,b):
        r=poses[:,:3,:3]
        if (not np.allclose(poses[:,3],[0,0,0,1],atol=1e-8)
                or not np.allclose(r@r.swapaxes(1,2),np.eye(3),atol=1e-5)
                or not np.allclose(np.linalg.det(r),1,atol=1e-5)):
            raise ValueError('Rigid homogeneous poses required')
    ra=Rotation.from_matrix(a[:,:3,:3]);delta=(Rotation.from_matrix(b[:,:3,:3])*ra.inv()).as_rotvec()
    angle=np.linalg.norm(delta,axis=1);limit=np.deg2rad(maximum_degrees)
    changed=angle>limit;out=b.copy()
    if changed.any():
        clipped=delta[changed]*(limit/angle[changed])[:,None]
        out[changed,:3,:3]=(Rotation.from_rotvec(clipped)*Rotation.from_matrix(a[changed,:3,:3])).as_matrix()
    return out,dict(schema='real2sim2real.geodesic-proposal-bound/1',maximum_degrees=float(maximum_degrees),
        changed_indices=np.flatnonzero(changed).tolist(),max_before_deg=float(np.rad2deg(angle.max())),
        max_after_deg=float(np.rad2deg((Rotation.from_matrix(out[:,:3,:3])*ra.inv()).magnitude()).max()),
        translations_unchanged=bool(np.array_equal(out[:,:3,3],b[:,:3,3])),
        proposal_accepted=False,requires_full_revalidation=True)
