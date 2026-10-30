"""Mesh-derived rigid-object inertials in metres and kilograms.

Uniform density over a closed reconstructed mesh is an explicit modelling
assumption, not an estimate of the real object's material or measured mass.
The visual/collision mesh and link coordinate frame remain unchanged.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def mass_properties(vertices, faces, *, mass_kg=1.):
    import numpy as np
    import trimesh
    if isinstance(mass_kg,bool) or not np.isfinite(mass_kg) or mass_kg<=0:
        raise ValueError('Positive finite explicit object mass required')
    vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
        or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
        or not len(faces) or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Finite metric vertices and triangle indices required')
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Mesh-derived volume inertials require a closed consistently wound mesh')
    if mesh.volume<0:mesh.invert()
    if not np.isfinite(mesh.volume) or mesh.volume<=1e-12:raise ValueError('Positive nondegenerate mesh volume required')
    center=np.asarray(mesh.center_mass);inertia=np.asarray(mesh.moment_inertia)*(float(mass_kg)/mesh.mass)
    eigen=np.linalg.eigvalsh(inertia)
    if not np.isfinite(center).all() or not np.isfinite(inertia).all() or eigen.min()<=0 or eigen.max()>eigen.sum()-eigen.max()+1e-10:
        raise ValueError('Physically valid positive rigid-body inertia required')
    return dict(mass_kg=float(mass_kg),center_of_mass_m=center.tolist(),inertia_kg_m2=inertia.tolist(),
        mesh_volume_m3=float(mesh.volume),assumption='Uniform density over the closed reconstructed volume; supplied nominal mass',
        actual_mass_measured=False,geometry_changed=False,link_frame_changed=False)


def write_urdf(mesh_path, output, *, mass_kg=1.):
    import json
    import numpy as np
    import trimesh
    from real2sim2real.artifacts import digest
    output=Path(output);source=(output.parent/mesh_path).resolve()
    mesh=trimesh.load(source,force='mesh',process=True)
    value=mass_properties(mesh.vertices,mesh.faces,mass_kg=mass_kg)
    root=ET.Element('robot',name='retarget_object');link=ET.SubElement(root,'link',name='object')
    inertial=ET.SubElement(link,'inertial')
    ET.SubElement(inertial,'origin',xyz=' '.join(format(x,'.12g') for x in value['center_of_mass_m']),rpy='0 0 0')
    ET.SubElement(inertial,'mass',value=format(value['mass_kg'],'.12g'))
    I=np.asarray(value['inertia_kg_m2']);ET.SubElement(inertial,'inertia',**{name:format(I[i,j],'.12g') for name,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})
    for role in ('visual','collision'):
        node=ET.SubElement(link,role);ET.SubElement(node,'origin',xyz='0 0 0',rpy='0 0 0')
        ET.SubElement(ET.SubElement(node,'geometry'),'mesh',filename=str(mesh_path))
    output.parent.mkdir(parents=True,exist_ok=True);ET.indent(root)
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    value.update(mesh_sha256=digest(source),urdf_sha256=digest(output),units='metres, kilograms, kilogram metres squared')
    output.with_suffix('.inertia.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return str(output.resolve())
