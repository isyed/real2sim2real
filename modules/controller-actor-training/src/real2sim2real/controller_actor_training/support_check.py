"""CPU initial-object placement against explicit convex triangle supports.

This measures the prepared simulator geometry. It does not establish the real
support's accuracy, complete triangle nonintersection, grasping or task success.
"""
from pathlib import Path
import argparse
import os
import xml.etree.ElementTree as ET


def plane_placement_metrics(points,plane_world,*,max_penetration_m=.02,max_gap_m=.05):
    """Exact vertex heights against an explicitly supplied infinite floor plane."""
    import numpy as np
    p,n=np.asarray(points,float),np.asarray(plane_world,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all()
            or n.shape!=(4,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n[:3]),1,atol=1e-6)
            or not np.isfinite([max_penetration_m,max_gap_m]).all() or min(max_penetration_m,max_gap_m)<0):
        raise ValueError('Finite metric vertices, unit explicit floor plane and nonnegative limits required')
    low=float((p@n[:3]+n[3]).min());penetration=max(0.,-low);gap=max(0.,low)
    return dict(passed=penetration<=max_penetration_m and gap<=max_gap_m,
        minimum_signed_height_m=low,maximum_sampled_penetration_m=penetration,minimum_surface_gap_m=gap,
        sampled_object_vertices=len(p),plane_world=n.tolist(),
        thresholds=dict(max_penetration_m=max_penetration_m,max_gap_m=max_gap_m),
        physical_execution_established=False,real_world_accuracy_established=False)


def placement_metrics(points, components, *, max_penetration_m=.02, max_gap_m=.05):
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty [N,3] object samples are required')
    if not np.isfinite([max_penetration_m,max_gap_m]).all() or min(max_penetration_m,max_gap_m)<0:
        raise ValueError('Placement limits must be finite and nonnegative')
    reports=[]
    for name,vertices,faces in components:
        vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=int)
        if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
                or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
                or faces.min()<0 or faces.max()>=len(vertices)):
            raise ValueError('Invalid support triangle mesh')
        mesh=trimesh.Trimesh(vertices,faces,process=False);hull=ConvexHull(vertices)
        if not mesh.is_watertight or not np.isclose(abs(mesh.volume),hull.volume,rtol=1e-4,atol=1e-9):
            raise ValueError('Placement half-space checks require closed convex supports')
        signed=np.max(points@hull.equations[:,:3].T+hull.equations[:,3],axis=1)
        penetration=max(0.,-float(signed.min()));gap=float('inf')
        for start in range(0,len(points),64):
            _,distance,_=trimesh.proximity.closest_point_naive(mesh,points[start:start+64])
            gap=min(gap,float(distance.min()))
        reports.append(dict(prim=name,vertices=len(vertices),faces=len(faces),
            minimum_surface_gap_m=gap,maximum_sampled_inside_depth_m=penetration))
    if not reports:raise ValueError('No explicit support triangle components')
    penetration=max(x['maximum_sampled_inside_depth_m'] for x in reports)
    gap=min(x['minimum_surface_gap_m'] for x in reports)
    return dict(passed=penetration<=max_penetration_m and gap<=max_gap_m,
        maximum_sampled_penetration_m=penetration,minimum_surface_gap_m=gap,
        sampled_object_vertices=len(points),components=reports,
        thresholds=dict(max_penetration_m=max_penetration_m,max_gap_m=max_gap_m))


def audit(reference, *, max_penetration_m=.02, max_gap_m=.05,floor_plane_world=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU placement checks')
    import numpy as np
    import pyarrow.parquet as pq
    import trimesh
    from scipy.spatial.transform import Rotation
    from real2sim2real.artifacts import digest
    from flash_chord.data import load_reference
    from flash_chord.assets.registry import support_usda_for_reference
    from .render_frozen import support_meshes
    reference=Path(reference).resolve();row=pq.read_table(reference).to_pylist()[0]
    ref=load_reference(reference,control_fps=50.,source_frame_playback=True)
    assets=ref.object_assets()
    if len(assets)!=1:raise ValueError('This placement audit requires one rigid object')
    urdf=Path(assets[0].urdf_path);pieces=[]
    for collision in ET.parse(urdf).getroot().findall('.//collision'):
        node=collision.find('geometry/mesh')
        if node is None:raise ValueError('This audit requires explicit object collision meshes')
        mesh=trimesh.load((urdf.parent/node.attrib['filename']).resolve(),force='mesh',process=False)
        vertices=np.asarray(mesh.vertices)*np.fromstring(node.get('scale','1 1 1'),sep=' ')
        origin=collision.find('origin')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        pieces.append(vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz)
    if not pieces:raise ValueError('Missing object collision geometry')
    local=np.concatenate(pieces)
    ids=np.unique(np.r_[np.linspace(0,len(local)-1,min(2048,len(local))).astype(int),local.argmin(0),local.argmax(0)])
    q=np.asarray(row['object_body_wxyz'])[0,0];p=np.asarray(row['object_body_position'])[0,0]
    points=local[ids]@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+p
    support=support_usda_for_reference(ref)
    if support is None:
        if floor_plane_world is None:raise ValueError('Explicit support geometry is required; floor-only scenes need an explicit floor_plane_world')
        all_points=local@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+p
        result=plane_placement_metrics(all_points,floor_plane_world,max_penetration_m=max_penetration_m,max_gap_m=max_gap_m)
        return dict(**result,scope='Initial object collision vertices against explicitly declared simulator floor',
            reference_sha256=digest(reference),support_sha256=None,urdf_sha256=digest(urdf),
            limitations=['The simulator floor plane is explicit, not inferred from a missing support mesh.',
                        'Floor clearance is not object stability or real-world calibration.'])
    if floor_plane_world is not None:raise ValueError('Choose explicit support meshes or floor-only placement, not both')
    components=[]
    for i,(_,triangles,_) in enumerate(support_meshes(support,['support'])):
        # Preserve the actual triangle surface; merge its repeated vertices.
        mesh=trimesh.Trimesh(triangles.reshape(-1,3),np.arange(len(triangles)*3).reshape(-1,3),process=True)
        components.append((str(i),mesh.vertices,mesh.faces))
    result=placement_metrics(points,components,max_penetration_m=max_penetration_m,max_gap_m=max_gap_m)
    return dict(**result,scope='Initial object placement against prepared simulator support colliders',
        reference_sha256=digest(reference),support_sha256=digest(support),urdf_sha256=digest(urdf),
        real_world_accuracy_established=False,physical_execution_established=False,
        limitations=['Sampled object vertices do not certify all triangle intersections.',
                    'Predicted support geometry may differ from the real surface.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior placement results')
    from real2sim2real.artifacts import write_json
    result=audit(a.reference);write_json(a.output,result)
    if not result['passed']:raise SystemExit('Initial placement exceeds the declared limits')
