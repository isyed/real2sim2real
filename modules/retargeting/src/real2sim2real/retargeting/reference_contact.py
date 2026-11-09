"""Audit reconstructed effector points against the actual object surface.

Geometric proximity is a reference diagnostic, not measured physical contact.
This check never edits motion, contact labels, collision geometry or thresholds.
"""
import numpy as np


def audit(vertices, faces, object_poses, hand_points, *, sides=('left', 'right'),
          threshold_m=0.02, maximum_candidates_per_point=100000):
    """Return per-hand frame counts using exact triangle distance near the gate.

    Poses are object-to-world [T,4,4]; points are world [T,H,P,3]. ``sides``
    labels the effectors: hands, feet or other explicitly selected body parts. All
    distances use metres. A conservative centroid/radius broad phase includes
    every triangle that could lie within the threshold, even for sparse meshes.
    Vertex distance alone cannot establish lack of surface proximity.
    """
    from scipy.spatial import cKDTree
    from trimesh.triangles import closest_point

    vertices = np.asarray(vertices, dtype=float)
    raw_faces = np.asarray(faces)
    poses = np.asarray(object_poses, dtype=float)
    points = np.asarray(hand_points, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,) or not len(raw_faces)
            or raw_faces.dtype.kind not in 'iu' or raw_faces.min() < 0 or raw_faces.max() >= len(vertices)):
        raise ValueError('Expected finite triangle mesh with integer indices')
    if (poses.ndim != 3 or poses.shape[1:] != (4,4) or not len(poses)
            or points.ndim != 4 or points.shape[:2] != (len(poses),len(sides))
            or points.shape[-1] != 3 or not points.shape[2] or len(set(sides)) != len(sides)):
        raise ValueError('Expected matching object poses and labelled hand points')
    if (not all(np.isfinite(x).all() for x in (vertices, poses, points))
            or not np.isfinite(threshold_m) or threshold_m <= 0
            or type(maximum_candidates_per_point) is not int or maximum_candidates_per_point < 1):
        raise ValueError('Require finite inputs, positive threshold and bounded search')
    rot = poses[:,:3,:3]
    if (not np.allclose(poses[:,3], [0,0,0,1], atol=1e-7)
            or not np.allclose(rot.swapaxes(1,2) @ rot, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rot), 1, atol=1e-4)):
        raise ValueError('Object transforms must be rigid and right handed')
    local = np.einsum('nji,nhpj->nhpi', rot, points-poses[:,None,None,:3,3])
    tri = vertices[raw_faces]
    centers = tri.mean(axis=1)
    radii = np.linalg.norm(tri-centers[:,None,:],axis=2).max(axis=1)
    max_radius = float(radii.max())
    center_tree = cKDTree(centers)
    flattened = local.reshape(-1,3)
    vertex_distance = cKDTree(vertices).query(flattened)[0].reshape(points.shape[:-1])
    contact = np.zeros(len(flattened), dtype=bool)
    for i, point in enumerate(flattened):
        ids = center_tree.query_ball_point(point, threshold_m+max_radius+1e-10)
        if len(ids) > maximum_candidates_per_point:
            raise ValueError('Triangle search budget exceeded; proximity remains unassessed')
        if not ids:
            continue
        ids = np.asarray(ids,dtype=int)
        ids = ids[np.linalg.norm(centers[ids]-point,axis=1) <= radii[ids]+threshold_m+1e-10]
        if len(ids):
            nearest = closest_point(tri[ids],np.broadcast_to(point,(len(ids),3)))
            contact[i] = bool((np.linalg.norm(nearest-point,axis=1) < threshold_m).any())
    contact = contact.reshape(points.shape[:-1]).any(axis=2)
    diameter = max(float(np.linalg.norm(tri[:,i]-tri[:,j],axis=1).max()) for i,j in ((0,1),(1,2),(2,0)))
    rows=[]
    for h,side in enumerate(sides):
        distance = vertex_distance[:,h].min(axis=1)
        rows.append(dict(side=side,frames_within_threshold=int(contact[:,h].sum()),
            active_frame_ids=np.flatnonzero(contact[:,h]).tolist(),
            minimum_vertex_distance_m=float(distance.min()),
            minimum_surface_distance_lower_bound_m=max(0.,float(distance.min())-diameter)))
    return dict(schema='real2sim2real.reference-contact-readiness/1',frames=len(poses),
        threshold_m=float(threshold_m),effectors=rows,hands=rows,any_proximity=bool(contact.any()),
        distance_method='Exact triangle distance within threshold; conservative centroid-radius broad phase',
        labels_modified=False,geometry_modified=False,physical_contact_established=False,
        actor_training_ready_established=False,
        recommendation='Proximity found; retargeting reach and unassisted physics still require validation' if contact.any()
        else 'No selected effector/object proximity: inspect geometry, alignment and task-specific source evidence before training')
