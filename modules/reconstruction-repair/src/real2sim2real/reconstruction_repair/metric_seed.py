"""Fresh CPU translation seeds after size calibration; never accepted motion.

An explicit orientation hypothesis is retained, but old object translations are
not inputs. Foreground surface depth is matched to front-facing mesh samples at
the same pixels, rather than treating surface depth as object-center depth.
"""
import numpy as np

from .temporal_pose import front_vertices, projected
from .geometry import mesh_silhouette


def assess_seed(vertices,faces,pose,intrinsics,object_mask,occluder_mask,depth_m,support_plane_camera):
    """Native image/depth assessment plus signed support clearance, no edits.

    Nonpenetration does not imply resting contact. A positive support gap is
    reported separately; whether contact is required belongs to the caller's
    declared phase/scene evidence. This never snaps an object onto a surface.
    """
    from .temporal_pose import measurements,TemporalPoseConfig
    v=np.asarray(vertices);f=np.asarray(faces);p=np.asarray(pose);K=np.asarray(intrinsics)
    m=np.asarray(object_mask);h=np.asarray(occluder_mask);z=np.asarray(depth_m);plane=np.asarray(support_plane_camera,float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or f.ndim!=2 or f.shape[1]!=3 or not len(f)
            or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v) or p.shape!=(4,4) or K.shape!=(3,3)
            or m.ndim!=2 or h.shape!=m.shape or z.shape!=m.shape or not np.isin(m,[0,1]).all() or not np.isin(h,[0,1]).all()
            or not all(np.isfinite(a).all() for a in [v,p,K,z]) or not np.allclose(p[3],[0,0,0,1])
            or not np.allclose(p[:3,:3]@p[:3,:3].T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(p[:3,:3]),1.,atol=1e-5)
            or not np.allclose(K[2],[0,0,1]) or min(K[0,0],K[1,1])<=0 or plane.shape!=(4,)
            or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-5)):
        raise ValueError('Compatible finite seed/camera/image/depth and unit support plane required')
    camera=v@p[:3,:3].T+p[:3,3];height=float(np.min(camera@plane[:3]+plane[3]))
    data=dict(vertices=v,faces=f,intrinsics=K[None],object_mask=m.astype(bool)[None],occluder_mask=h.astype(bool)[None],
        depth_m=z[None],frame_ids=np.array([0]),fit=np.array([True]),support_plane_camera=plane[None],
        support_center_camera=np.array([-plane[3]*plane[:3]]),support_radius_m=np.array([1e6]))
    result=measurements(data,p[None],TemporalPoseConfig())[0]
    result.update(minimum_support_height_m=height,support_gap_m=max(0.,height),
        nonpenetration_within_2cm=bool(height>=-.02),resting_clearance_within_2cm=bool(abs(height)<=.02),
        support_contact_requirement_not_inferred=True,poses_modified=False,physics_validated=False,approved_for_use=False)
    return result


def compare_seeds(parent_vertices,parent_faces,parent_pose,proposal_vertices,proposal_faces,proposal_pose,
                  intrinsics,object_mask,occluder_mask,depth_m,support_plane_camera,*,max_iou_drop=.02,require_resting=False):
    """Recompute actual parent/proposal on identical pixels before promotion.

    Comparing two newly registered alternatives can miss degradation of a good
    input pose. Size-proxy agreement alone is insufficient to replace that input.
    Resting support is a declared phase assumption, never inferred from a gap.
    """
    if not np.isfinite(max_iou_drop) or not 0<=max_iou_drop<=1 or type(require_resting) is not bool:
        raise ValueError('Explicit image tolerance and resting-phase boolean required')
    args=(intrinsics,object_mask,occluder_mask,depth_m,support_plane_camera)
    before=assess_seed(parent_vertices,parent_faces,parent_pose,*args)
    after=assess_seed(proposal_vertices,proposal_faces,proposal_pose,*args)
    reasons=[]
    if before['iou']>=.6 and after['iou']<before['iou']-max_iou_drop:reasons.append('previous_good_image_regressed')
    if min(after['iou'],after['precision'],after['recall'])<.6:reasons.append('proposal_image_gate')
    if after['depth_points']<100 or after['depth_error_m'] is None or after['depth_error_m']>.1:reasons.append('proposal_depth_gate')
    if after['behind_camera']:reasons.append('proposal_camera_gate')
    if not after['nonpenetration_within_2cm']:reasons.append('proposal_support_penetration')
    if require_resting and not after['resting_clearance_within_2cm']:reasons.append('proposal_resting_support_gap')
    return dict(schema='r2s2r_engines.metric-seed-parent-comparison/1',before=before,after=after,
        same_pixel_evidence=True,max_iou_drop=max_iou_drop,resting_assumption_declared=require_resting,
        proposal_passed=not reasons,reasons=reasons,iou_delta=after['iou']-before['iou'],
        depth_error_delta_m=(after['depth_error_m']-before['depth_error_m']) if after['depth_error_m'] is not None and before['depth_error_m'] is not None else None,
        poses_modified=False,full_reference_accepted=False,actor_training_approved=False,approved_for_use=False)


def rotation_from_mesh_prediction(quaternion_wxyz, source_to_aligned):
    """Express an RGB-predicted rotation in the exported aligned mesh frame.

    Quaternion convention is explicit; callers must not guess xyzw versus wxyz.
    The provided transform maps source mesh coordinates to exported coordinates.
    Translation/scale predictions are intentionally not transferred.
    """
    from scipy.spatial.transform import Rotation
    q, transform = np.asarray(quaternion_wxyz, float), np.asarray(source_to_aligned, float)
    if (q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.) > 1e-4
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Explicit unit wxyz quaternion and rigid source-to-aligned transform are required')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ transform[:3, :3].T


def register_translation(vertices, faces, rotation, intrinsics, object_mask,
                         occluder_mask, depth_m, *, iterations=8):
    v, f, R, K = (np.asarray(x) for x in (vertices, faces, rotation, intrinsics))
    mask, excluded, depth = (np.asarray(x) for x in (object_mask, occluder_mask, depth_m))
    if (v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3
            or f.dtype.kind not in 'iu' or not len(f) or f.min() < 0 or f.max() >= len(v)
            or R.shape != (3, 3) or K.shape != (3, 3)
            or not all(np.isfinite(a).all() for a in (v, R, K, depth))
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-5)
            or not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0
            or mask.ndim != 2 or mask.shape != excluded.shape or mask.shape != depth.shape
            or not np.isin(mask, [0, 1]).all() or not np.isin(excluded, [0, 1]).all()
            or type(iterations) is not int or not 1 <= iterations <= 16):
        raise ValueError('Invalid fresh metric registration inputs')
    observed = mask.astype(bool) & ~excluded.astype(bool)
    valid_depth = observed & (depth > 0) & (depth < 65.535)
    if valid_depth.sum() < 100:
        raise ValueError('At least 100 visible object depth pixels are required')
    y, x = np.where(observed)
    target = np.array([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2])
    rotated = v @ R.T
    t = np.linalg.solve(K, np.r_[target, 1.]) * np.median(depth[valid_depth])
    t -= (rotated.min(0) + rotated.max(0)) / 2
    pose = np.eye(4); pose[:3, :3] = R
    last_count = 0
    for _ in range(iterations):
        pose[:3, 3] = t
        if (rotated + t)[:, 2].min() <= .01:
            raise ValueError('Registration crosses the camera near plane')
        _, uv = projected(v, pose, K)
        center = (uv.min(0) + uv.max(0)) / 2
        t[:2] += (target - center) * t[2] / K.diagonal()[:2]
        pose[:3, 3] = t
        points = front_vertices(v, f, pose)
        xyz, uv = projected(points, pose, K)
        xy = np.rint(uv).astype(int); h, w = observed.shape
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        flat = xy[keep, 1] * w + xy[keep, 0]
        unique, inverse = np.unique(flat, return_inverse=True)
        front = np.full(len(unique), np.inf)
        np.minimum.at(front, inverse, xyz[keep, 2])
        use = valid_depth.ravel()[unique]
        last_count = int(use.sum())
        if last_count < 100:
            raise ValueError('Insufficient corresponding foreground surface depth')
        t[2] += np.clip(np.median(depth.ravel()[unique[use]] - front[use]), -.20, .20)
    pose[:3, 3] = t
    if (rotated + t)[:, 2].min() <= .01:
        raise ValueError('Final seed crosses the camera near plane')
    g = dict(vertices=v, faces=f, object_scale=np.array(1.), object_rotation=R[None],
             object_translation=t[None], intrinsics=K[None], world_to_camera=np.eye(4)[None])
    rendered = mesh_silhouette(g, 0, *observed.shape) & ~excluded.astype(bool)
    intersection = int((rendered & observed).sum())
    metrics = dict(iou=intersection / max(1, int((rendered | observed).sum())),
                   precision=intersection / max(1, int(rendered.sum())),
                   recall=intersection / max(1, int(observed.sum())),
                   depth_correspondences=last_count, orientation_refitted=False,
                   old_translation_used=False, trajectory_accepted=False,
                   physics_validated=False, approved_for_use=False)
    return pose, metrics


def seed_from_calibration(calibration, source, provenance_path, output):
    """Hash-verified CLI boundary. A seed is never a replacement trajectory."""
    import json
    from pathlib import Path
    from .bundle import arrays_atomic, arrays_read, file_sha256, json_atomic
    calibration, source, output = map(Path, (calibration, source, output))
    report = json.loads((calibration / 'calibration.json').read_text())
    provenance = json.loads(Path(provenance_path).read_text())
    mesh_path = calibration / 'calibrated_mesh.npz'
    if (not report.get('candidate_ready') or not report.get('requires_retracking')
            or report.get('old_poses_valid_for_resized_mesh') is not False
            or file_sha256(mesh_path) != report['mesh_sha256']
            or provenance.get('schema') != 'r2s2r_engines.metric-seed-evidence/1'
            or not provenance.get('source') or provenance.get('clean_trajectory_used') is not False
            or provenance.get('input_sha256') != file_sha256(source)
            or provenance.get('calibration_sha256') != file_sha256(calibration / 'calibration.json')):
        raise ValueError('A verified size hypothesis and explicit prediction-only seed provenance are required')
    mesh = arrays_read(mesh_path); data = arrays_read(source)
    if set(data) != {'rotation', 'intrinsics', 'object_mask', 'occluder_mask', 'depth_m'}:
        raise ValueError('Seed input fields differ from the explicit contract')
    if output.exists():
        raise ValueError('Preserve previous seed outputs; use a new directory')
    pose, metrics = register_translation(mesh['vertices'], mesh['faces'], **data)
    output.mkdir(parents=True)
    arrays_atomic(output / 'seed.npz', {'pose_camera': pose})
    json_atomic(output / 'report.json', dict(**metrics, provenance=provenance,
        limitations=['Orientation is an explicit hypothesis and is not refitted.',
                     'Image/depth agreement cannot establish interaction or physical validity.',
                     'Fresh tracking and independent full-clip validation remain mandatory.']))
    return metrics
