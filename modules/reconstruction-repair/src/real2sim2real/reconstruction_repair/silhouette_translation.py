"""Fast fixed-rotation translation proposals from a visible image bounding box.

Bounding-box fit is initialization only: occlusion and wrong orientation can
produce misleading proposals. Native silhouette, depth, support, temporal and
independent renderer checks remain the caller's responsibility.
"""
import numpy as np


def projected_box(vertices,rotation,translation,intrinsics):
    points=np.asarray(vertices)@np.asarray(rotation).T+np.asarray(translation)
    if np.any(points[:,2]<=.01):raise ValueError('Mesh crosses the camera plane')
    image=points@np.asarray(intrinsics).T;image=image[:,:2]/image[:,2:]
    return np.r_[image.min(0),image.max(0)]


def propose(vertices,rotation,translation,intrinsics,box_xyxy,*,max_shift_m=.25,max_calls=60,preserve_camera_depth=False):
    from scipy.optimize import least_squares
    v,R,t,K,b=map(lambda x:np.asarray(x,float),(vertices,rotation,translation,intrinsics,box_xyxy))
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or R.shape!=(3,3) or t.shape!=(3,)
            or K.shape!=(3,3) or b.shape!=(4,) or not all(np.isfinite(x).all() for x in (v,R,t,K,b))
            or not np.allclose(R@R.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5)
            or not np.allclose(K[2],[0,0,1]) or min(K[0,0],K[1,1])<=0
            or np.any(b[:2]<0) or np.any(b[2:]<=b[:2])
            or isinstance(max_shift_m,bool) or not np.isfinite(max_shift_m) or max_shift_m<=0
            or type(max_calls) is not int or max_calls<1 or type(preserve_camera_depth) is not bool):
        raise ValueError('Finite metric mesh, rigid rotation, camera, nonempty image box and explicit bounds required')
    before=projected_box(v,R,t,K)
    def displacement(parameters):return np.r_[parameters,0.] if preserve_camera_depth else parameters
    def residual(parameters):
        delta=displacement(parameters)
        try:box=projected_box(v,R,t+delta,K)
        except ValueError:return np.full(5,1e6)
        return np.r_[box-b,1000*max(0.,np.linalg.norm(delta)-max_shift_m)]
    fitted=least_squares(residual,np.zeros(2 if preserve_camera_depth else 3),bounds=(-max_shift_m,max_shift_m),
        max_nfev=max_calls,loss='soft_l1',diff_step=1e-4)
    shift=displacement(fitted.x);candidate=t+shift
    after=projected_box(v,R,candidate,K)
    return candidate,dict(schema='real2sim2real.silhouette-translation-proposal/1',
        bounding_box_error_before_px=float(np.linalg.norm(before-b)),
        bounding_box_error_after_px=float(np.linalg.norm(after-b)),
        translation_change_m=float(np.linalg.norm(shift)),
        within_translation_bound=bool(np.linalg.norm(shift)<=max_shift_m+1e-7),
        camera_depth_preserved=bool(candidate[2]==t[2]),
        rotation_changed=False,mesh_changed=False,depth_used_for_fitting=False,
        proposal_only=True,reference_accepted=False,physics_validated=False,
        full_silhouette_validation_required=True,independent_depth_validation_required=True)
