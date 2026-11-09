"""Bounded temporal pose priors; every output still needs source validation.

Regularize camera-space object centres and rotation corrections relative to a
reference sequence. This avoids averaging matrix entries or confusing a smooth
trajectory with measured evidence. No image, depth or physics gate is implied.
"""
import numpy as np
from scipy.sparse import diags,eye
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation


def propose(reference,observed,timestamps_s,*,local_center,locked,
            rotation_time_s=.10,translation_time_s=.10,
            max_rotation_correction_deg=45.,max_center_shift_m=.40):
    a,b,t,c=np.asarray(reference,float),np.asarray(observed,float),np.asarray(timestamps_s,float),np.asarray(local_center,float)
    lock=np.asarray(locked)
    n=len(a)
    limits=np.asarray([rotation_time_s,translation_time_s,max_rotation_correction_deg,max_center_shift_m],float)
    if (a.shape!=(n,4,4) or b.shape!=a.shape or n<5 or t.shape!=(n,) or c.shape!=(3,)
            or lock.shape!=(n,) or lock.dtype!=bool or not lock.any()
            or not all(np.isfinite(v).all() for v in (a,b,t,c,limits))
            or np.any(np.diff(t)<=0) or np.any(limits<=0) or max_rotation_correction_deg>=90):
        raise ValueError('Matching finite native poses/times, metric centre, explicit anchors and positive bounds required')
    for p in (a,b):
        if (not np.allclose(p[:,3],[0,0,0,1]) or not np.allclose(p[:,:3,:3]@p[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-5)
                or not np.allclose(np.linalg.det(p[:,:3,:3]),1,atol=1e-5)):
            raise ValueError('Rigid pose matrices required')
    dt=np.diff(t)
    if not np.allclose(dt,dt[0],rtol=1e-5,atol=1e-9):raise ValueError('Uniform native source clock required')
    correction=(Rotation.from_matrix(b[:,:3,:3])*Rotation.from_matrix(a[:,:3,:3]).inv()).as_rotvec()
    if np.linalg.norm(correction,axis=1).max()>=np.pi/2:raise ValueError('Local rotation chart exceeds the declared small-correction domain')
    ac=np.einsum('nij,j->ni',a[:,:3,:3],c)+a[:,:3,3]
    bc=np.einsum('nij,j->ni',b[:,:3,:3],c)+b[:,:3,3]
    d=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    edit=np.flatnonzero(~lock)
    def smooth(values,seconds):
        # Anchor corrections are zero: the returned locked poses equal reference.
        matrix=eye(n,format='csc')+(seconds/dt[0])**4*(d.T@d)
        result=np.zeros_like(values)
        if len(edit):result[edit]=spsolve(matrix[edit][:,edit],values[edit])
        return result
    rc=smooth(correction,rotation_time_s);tc=smooth(bc-ac,translation_time_s)
    degrees=np.rad2deg(np.linalg.norm(rc,axis=1));shift=np.linalg.norm(tc,axis=1)
    if degrees.max()>max_rotation_correction_deg or shift.max()>max_center_shift_m:
        raise ValueError('Regularized prior exceeds the declared source uncertainty')
    result=a.copy();result[:,:3,:3]=(Rotation.from_rotvec(rc)*Rotation.from_matrix(a[:,:3,:3])).as_matrix()
    result[:,:3,3]=ac+tc-np.einsum('nij,j->ni',result[:,:3,:3],c)
    result[lock]=a[lock]
    return result,dict(schema='real2sim2real.temporal-pose-prior/1',proposal_only=True,
        rotation_time_s=rotation_time_s,translation_time_s=translation_time_s,
        max_rotation_correction_deg=float(degrees.max()),max_center_shift_m=float(shift.max()),
        locked_frames=np.flatnonzero(lock).tolist(),locked_poses_preserved=True,
        input_frame_count=n,output_frame_count=n,missing_frames_filled=False,
        source_evidence_validated=False,full_reference_approved=False,physics_validated=False,
        limitation='Smooth prior only: independently measure native image/depth/support and temporal gates')
