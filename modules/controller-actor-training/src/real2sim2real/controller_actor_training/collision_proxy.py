"""Explicit convex collision approximation for a portable robot reference.

The source visual mesh, motion, inertial parameters and support are preserved.
A convex proxy fills concavities: this is a declared simulation hypothesis, not
a geometry repair or a default collision setting.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from real2sim2real.artifacts import load,publish,digest,write_json
from .frozen import copy_tree


def package(source,output,*,urdf_role,hardlink=False):
    import numpy as np
    import trimesh
    source,output=map(lambda p:Path(p).resolve(),[source,output])
    record=load(source,kinds={'robot-reference'})
    if output.exists() or output.is_relative_to(source):raise ValueError('Use a fresh independent output')
    path=record['files'][urdf_role]['path'];urdf=source/path
    tree=ET.parse(urdf);root=tree.getroot()
    collision=root.findall('.//collision');visual=root.findall('.//visual')
    if len(collision)!=1 or len(visual)!=1:raise ValueError('Exactly one object mesh collision and visual required')
    c,v=collision[0].find('geometry/mesh'),visual[0].find('geometry/mesh')
    if c is None or v is None or c.attrib!=v.attrib:raise ValueError('Source collision and visual mesh must match')
    def origin(node):
        value=node.find('origin');return value.attrib if value is not None else {}
    if origin(collision[0])!=origin(visual[0]):raise ValueError('Source collision and visual origins differ')
    mesh_path=(urdf.parent/v.get('filename')).resolve()
    if not mesh_path.is_relative_to(source):raise ValueError('Mesh escaped the portable source')
    mesh=trimesh.load(mesh_path,force='mesh',process=False);hull=mesh.convex_hull
    if not hull.is_watertight or not np.allclose(hull.bounds,mesh.bounds,atol=1e-6,rtol=0):
        raise ValueError('Closed proxy with unchanged metric bounds required')
    if len(hull.faces)>=len(mesh.faces):raise ValueError('Convex proxy does not simplify the source mesh')
    inertial=ET.tostring(root.find('.//inertial'))
    copy_tree(source,output,hardlink=hardlink)
    proxy=(output/path).parent/'convex_collision_proxy.obj'
    if proxy.exists():raise FileExistsError('Preserve existing proxy')
    hull.export(proxy,include_color=False,include_texture=False)
    c.set('filename',proxy.name)
    modified=output/path;temporary=modified.with_suffix('.pending.urdf')
    tree.write(temporary,encoding='utf-8',xml_declaration=True);temporary.replace(modified)
    if ET.tostring(ET.parse(modified).getroot().find('.//inertial'))!=inertial:
        raise ValueError('Collision-only proposal changed inertial parameters')
    receipt=dict(source_reference_sha256=digest(source/'artifact.json'),visual_mesh_sha256=digest(mesh_path),
        source_faces=len(mesh.faces),proxy_faces=len(hull.faces),metric_bounds_unchanged=True,
        visual_geometry_changed=False,inertial_parameters_changed=False,motion_changed=False,
        convex_proxy_fills_concavities=True,physics_validated=False)
    write_json(output/'collision_proxy.json',receipt)
    (output/'artifact.json').unlink()
    files={key:output/value['path'] for key,value in record['files'].items()}
    files.update(collision_proxy=proxy,collision_proxy_receipt=output/'collision_proxy.json')
    publish(output,'robot-reference',files,metadata=dict(record['metadata'],physics_validated=False,
        collision_proxy='Explicit convex collision approximation; visual and inertia unchanged',
        parent_reference_sha256=receipt['source_reference_sha256']))
    load(source,kinds={'robot-reference'});return receipt
