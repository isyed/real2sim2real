"""Bounded collision-only hypothesis for a noisy reconstructed resting base.

The caller must supply a reviewed support normal in object coordinates and an
explicit cut height. This never changes visual geometry, reference poses, mass
properties or a selected artifact. A proposal still requires image/placement and
isolated CPU/GPU settling checks before adoption.
"""
import numpy as np


def propose(vertices,faces,normal,*,cut_above_lowest_m,max_correction_m=.01,
            max_volume_loss_fraction=.05):
    import trimesh
    v=np.asarray(vertices,float);f=np.asarray(faces);n=np.asarray(normal,float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
        or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not f.size
        or f.min()<0 or f.max()>=len(v) or n.shape!=(3,) or not np.isfinite(n).all()
        or not np.isclose(np.linalg.norm(n),1.,atol=1e-6)):
        raise ValueError('Finite metric mesh and unit object-frame support normal required')
    if (not np.isfinite([cut_above_lowest_m,max_correction_m,max_volume_loss_fraction]).all()
        or not 0<cut_above_lowest_m<=max_correction_m or max_correction_m<=0
        or not 0<max_volume_loss_fraction<1):
        raise ValueError('Declare bounded positive metric correction and volume-loss limit')
    original=trimesh.Trimesh(v,f,process=False)
    if not original.is_watertight or not original.is_convex or original.volume<=0:
        raise ValueError('This hypothesis requires a closed convex collision proxy')
    height=v@n;level=float(height.min()+cut_above_lowest_m)
    move=np.maximum(level-height,0.)
    corrected=v+move[:,None]*n
    hull=trimesh.Trimesh(corrected,f,process=False).convex_hull
    loss=1-float(hull.volume/original.volume)
    if not -1e-7<=loss<=max_volume_loss_fraction:
        raise ValueError('Collision hypothesis exceeds volume-loss bound')
    # Each old point moves only along n; tangential coordinates are preserved.
    if np.max(move)>max_correction_m+1e-10:raise ValueError('Metric bound violated')
    return np.asarray(hull.vertices),np.asarray(hull.faces),dict(
        schema='real2sim2real.planar-collision-base/1',
        original_vertices=len(v),original_faces=len(f),candidate_faces=len(hull.faces),
        changed_input_vertices=int(np.count_nonzero(move)),
        normal_object=n.tolist(),original_lowest_m=float(height.min()),candidate_lowest_m=level,
        max_vertex_correction_m=float(move.max()),max_correction_m=float(max_correction_m),
        volume_loss_fraction=loss,max_volume_loss_fraction=float(max_volume_loss_fraction),
        visual_geometry_changed=False,reference_pose_changed=False,
        mass_properties_changed=False,proposal_only=True,physics_validated=False,
        resting_shape_established=False,selected_reference_changed=False)
