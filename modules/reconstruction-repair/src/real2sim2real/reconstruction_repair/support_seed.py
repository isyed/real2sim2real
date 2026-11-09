"""Explicit, bounded resting-support seed hypotheses; never trajectory edits."""
import numpy as np
from .metric_seed import assess_seed, compare_seeds


def resting_orientation_hypotheses(vertices, pose, center_local, support_plane, *,
                                   resting_evidence, max_rotation_deg=30., max_candidates=8):
    """Nearest convex-support-face orientations, retaining the world centre.

    Each candidate's projected centre lies inside its proposed support triangle.
    This is a quasi-static geometric hypothesis, not a collision/settling proof.
    No contact translation is added: callers must measure support clearance and
    decide whether a drop is admissible. No input arrays or trajectory are edited.
    """
    from scipy.spatial import ConvexHull,QhullError
    from scipy.spatial.transform import Rotation
    from .geometry import rotation_between
    v=np.asarray(vertices,float);p=np.asarray(pose,float);center=np.asarray(center_local,float)
    plane=np.asarray(support_plane,float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or p.shape!=(4,4) or center.shape!=(3,)
            or plane.shape!=(4,) or not all(np.isfinite(x).all() for x in [v,p,center,plane])
            or not np.allclose(p[3],[0,0,0,1]) or not np.allclose(p[:3,:3]@p[:3,:3].T,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(p[:3,:3]),1.,atol=1e-6)
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6)
            or not isinstance(resting_evidence,str) or not resting_evidence.strip()
            or isinstance(max_rotation_deg,bool) or not np.isfinite(max_rotation_deg) or not 0<max_rotation_deg<=45
            or type(max_candidates) is not int or not 1<=max_candidates<=32):
        raise ValueError('Finite metric geometry, rigid pose, unit support plane and bounded resting evidence required')
    try:hull=ConvexHull(v)
    except QhullError as error:raise ValueError('Resting hypothesis requires a full-dimensional mesh') from error
    if np.any(hull.equations[:,:3]@center+hull.equations[:,3]>1e-7):
        raise ValueError('Declared centre lies outside the mesh convex hull')
    world_center=p[:3,:3]@center+p[:3,3];rows=[]
    for index,(face,eq) in enumerate(zip(hull.simplices,hull.equations)):
        triangle=v[face];n=eq[:3];point=center-((center-triangle[0])@n)*n
        a,b=triangle[1]-triangle[0],triangle[2]-triangle[0];q=point-triangle[0]
        aa,ab,bb=a@a,a@b,b@b;det=aa*bb-ab*ab
        if det<1e-20:continue
        x=(bb*(q@a)-ab*(q@b))/det;y=(aa*(q@b)-ab*(q@a))/det
        if min(x,y,1-x-y)<-1e-6:continue
        change=rotation_between(p[:3,:3]@n,-plane[:3]);angle=float(np.rad2deg(Rotation.from_matrix(change).magnitude()))
        if angle>max_rotation_deg+1e-8:continue
        proposal=p.copy();proposal[:3,:3]=change@p[:3,:3];proposal[:3,3]=world_center-proposal[:3,:3]@center
        if any(np.allclose(proposal,row['pose'],atol=1e-7) for row in rows):continue
        rows.append(dict(pose=proposal,rotation_deg=angle,convex_face_index=index,
            source_vertex_ids=face.tolist(),projected_center_barycentric=[float(1-x-y),float(x),float(y)],
            minimum_support_height_m=float(((v@proposal[:3,:3].T+proposal[:3,3])@plane[:3]+plane[3]).min())))
    rows.sort(key=lambda row:row['rotation_deg']);selected=rows[:max_candidates]
    return selected,dict(schema='real2sim2real.resting-orientation-hypotheses/1',
        candidates=len(selected),resting_evidence=resting_evidence,world_center_preserved=True,
        translation_to_support_added=False,geometry_modified=False,source_acceptance_required=True,
        physical_stability_established=False,actor_training_approved=False)


def local_rest_orientation(poses, frame_ids, rotation_local, center_local, *,
                           hold_end, fade_end, resting_evidence, max_rotation_deg=10.):
    """Propose a bounded initial-rest correction, then preserve later poses.

    Rotation acts about the caller's mesh-local centre, so object-centre motion
    is preserved. The explicit source-frame window is never inferred from hand
    labels or task scores. Image/depth, transition and physics gates remain the
    caller's responsibility; this function never approves a reference.
    """
    from scipy.spatial.transform import Rotation
    from .continuation import _poses
    parent,ids=_poses(poses,frame_ids)
    r=np.asarray(rotation_local,float);center=np.asarray(center_local,float)
    if (r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r@r.T,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(r),1.,atol=1e-6) or center.shape!=(3,)
            or not np.isfinite(center).all() or not isinstance(resting_evidence,str) or not resting_evidence.strip()
            or type(hold_end) is not int or type(fade_end) is not int
            or not ids[0]<=hold_end<fade_end<=ids[-1]
            or isinstance(max_rotation_deg,bool) or not np.isfinite(max_rotation_deg)
            or not 0<max_rotation_deg<=30):
        raise ValueError('Rigid local rotation, metric centre and explicit bounded resting window required')
    vector=Rotation.from_matrix(r).as_rotvec();angle=float(np.rad2deg(np.linalg.norm(vector)))
    if angle>max_rotation_deg+1e-8:raise ValueError('Resting rotation exceeds declared bound')
    weights=np.where(ids<=hold_end,1.,np.where(ids>=fade_end,0.,
        .5+.5*np.cos(np.pi*(ids-hold_end)/(fade_end-hold_end))))
    changed=np.flatnonzero(weights>0);out=parent.copy()
    for i in changed:
        correction=np.eye(4);correction[:3,:3]=Rotation.from_rotvec(vector*weights[i]).as_matrix()
        correction[:3,3]=center-correction[:3,:3]@center
        out[i]=parent[i]@correction
    before=parent[:,:3,:3]@center+parent[:,:3,3];after=out[:,:3,:3]@center+out[:,:3,3]
    return out,dict(schema='real2sim2real.local-rest-orientation/1',rotation_deg=angle,
        hold_end=hold_end,fade_end=fade_end,changed_frame_ids=ids[changed].tolist(),
        resting_evidence=resting_evidence,maximum_center_motion_error_m=float(np.max(np.abs(before-after))),
        later_poses_preserved_exactly=bool(np.array_equal(out[ids>=fade_end],parent[ids>=fade_end])),
        geometry_modified=False,scale_modified=False,proposal_only=True,
        image_depth_acceptance_required=True,physics_validated=False,actor_training_approved=False)


def propose_resting_seed(parent_vertices, parent_faces, parent_pose,
                         vertices, faces, seed, intrinsics, object_mask,
                         occluder_mask, depth_m, plane, *, resting_evidence,
                         max_translation_m=0.05, steps=5):
    """Try the smallest plane-normal move that passes parent/image/depth gates.

    The caller must provide reviewed evidence that this individual frame is a
    resting phase. No contact phase is inferred or propagated to other frames.
    A returned proposal still requires fresh tracking and physical validation.
    On rejection return the original seed unchanged, with every trial retained.
    """
    if (not isinstance(resting_evidence, str) or not resting_evidence.strip()
            or isinstance(max_translation_m, bool)
            or not np.isfinite(max_translation_m) or not 0 < max_translation_m <= .1
            or type(steps) is not int or not 1 <= steps <= 10):
        raise ValueError('Reviewed resting evidence and bounded search are required')
    initial = assess_seed(vertices, faces, seed, intrinsics, object_mask,
                          occluder_mask, depth_m, plane)
    gap = initial['minimum_support_height_m']
    direction = -np.sign(gap) * np.asarray(plane)[:3]
    moves = np.linspace(0., min(abs(gap), max_translation_m), steps + 1)
    trials = []
    selected = None
    for distance in moves:
        proposal = np.asarray(seed, float).copy()
        proposal[:3, 3] += direction * distance
        check = compare_seeds(parent_vertices, parent_faces, parent_pose,
                              vertices, faces, proposal, intrinsics, object_mask,
                              occluder_mask, depth_m, plane, require_resting=True)
        trials.append(dict(translation_m=float(distance), **check))
        if check['proposal_passed']:
            selected = proposal
            break
    return (np.asarray(seed).copy() if selected is None else selected), dict(
        schema='real2sim2real.resting-seed-proposal/1',
        candidate_available=selected is not None, trials=trials,
        resting_evidence=resting_evidence, maximum_translation_m=max_translation_m,
        original_seed_preserved_on_rejection=True, parent_modified=False,
        phase_propagated=False, requires_fresh_tracking=True,
        actor_training_approved=False, physical_contact_verified=False)
