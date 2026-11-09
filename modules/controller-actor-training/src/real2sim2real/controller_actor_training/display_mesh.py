"""Conservative display-only mesh reduction with a bounds-preservation guard.

Some decimators return severely truncated meshes for nearly coplanar geometry.
Never silently display those results as the source object. This is a coarse
extent guard, not certification of every thin feature or collision geometry.
"""
def simplify(mesh, face_budget, *, relative_bounds_tolerance=.02):
    import numpy as np
    if type(face_budget) is not int or face_budget<4 or not 0<relative_bounds_tolerance<.1:
        raise ValueError('Explicit positive display budget and bounds tolerance required')
    source=np.asarray(mesh.vertices);bounds=np.stack([source.min(0),source.max(0)])
    tolerance=max(1e-6,float(np.linalg.norm(bounds[1]-bounds[0]))*relative_bounds_tolerance)
    attempts=[]
    if len(mesh.faces)<=face_budget:return mesh,dict(method='source_mesh',source_faces=len(mesh.faces),display_faces=len(mesh.faces),attempts=[])
    for budget in dict.fromkeys([face_budget,max(30000,face_budget*10)]):
        if budget>=len(mesh.faces):break
        candidate=mesh.simplify_quadric_decimation(face_count=budget)
        v=np.asarray(candidate.vertices)
        valid=v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all() and len(candidate.faces)>0
        error=float(np.abs(np.stack([v.min(0),v.max(0)])-bounds).max()) if valid else None
        passed=bool(valid and error<=tolerance and len(candidate.faces)<=budget)
        attempts.append(dict(requested_faces=budget,returned_faces=len(candidate.faces),maximum_bounds_error=error,passed=passed))
        if passed:return candidate,dict(method='guarded_decimation',source_faces=len(mesh.faces),display_faces=len(candidate.faces),
            bounds_tolerance=tolerance,attempts=attempts,source_bounds=bounds.tolist(),physics_geometry_changed=False)
    return mesh,dict(method='source_fallback',source_faces=len(mesh.faces),display_faces=len(mesh.faces),
        bounds_tolerance=tolerance,attempts=attempts,source_bounds=bounds.tolist(),physics_geometry_changed=False)
