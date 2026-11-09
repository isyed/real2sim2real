"""Mesh-derived rigid-object inertials in metres and kilograms.

Uniform density over a closed reconstructed mesh is an explicit modelling
assumption, not an estimate of the real object's material or measured mass.
The visual/collision mesh and link coordinate frame remain unchanged.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def audit_inertial_pair(original, candidate):
    """Verify a rigid-object ablation changes inertials only, with fixed mass.

    Both visual/collision mesh bytes and all non-inertial XML must match. This
    audit does not establish physical stability or compatibility with an actor.
    """
    import numpy as np
    from real2sim2real.artifacts import digest
    paths=list(map(Path,[original,candidate]));trees=[ET.parse(p).getroot() for p in paths]
    properties=[];mesh_hashes=[]
    for path,tree in zip(paths,trees):
        links=tree.findall('link')
        if len(links)!=1 or tree.findall('joint'):raise ValueError('Require one rigid object link')
        inertials=links[0].findall('inertial')
        if len(inertials)!=1:raise ValueError('Require explicit object inertial')
        node=inertials[0];mass=float(node.find('mass').get('value'))
        origin=node.find('origin');com=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
        I=node.find('inertia');tensor=np.array([[float(I.get(k)) for k in row] for row in
            [('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz')]])
        if (not np.isfinite(mass) or mass<=0 or com.shape!=(3,) or rpy.shape!=(3,)
                or not np.isfinite(com).all() or not np.isfinite(rpy).all() or not np.isfinite(tensor).all()):
            raise ValueError('Invalid inertial values')
        eigen=np.linalg.eigvalsh(tensor)
        if eigen.min()<=0 or eigen.max()>eigen.sum()-eigen.max()+1e-10:raise ValueError('Invalid rigid-body inertia')
        properties.append(dict(mass_kg=mass,center_of_mass_m=com.tolist(),inertial_frame_rpy=rpy.tolist(),inertia_kg_m2=tensor.tolist()))
        links[0].remove(node)
        mesh_hashes.append([digest((path.parent/m.get('filename')).resolve()) for m in tree.iter('mesh')])
    def signature(node):return (node.tag,sorted(node.attrib.items()),(node.text or '').strip(),[signature(c) for c in node])
    if signature(trees[0])!=signature(trees[1]) or mesh_hashes[0]!=mesh_hashes[1]:
        raise ValueError('Non-inertial geometry or model differs')
    if properties[0]['mass_kg']!=properties[1]['mass_kg']:raise ValueError('Mass changed')
    return dict(original_sha256=digest(paths[0]),candidate_sha256=digest(paths[1]),
        original=properties[0],candidate=properties[1],mesh_sha256=mesh_hashes[0],
        geometry_unchanged=True,mass_unchanged=True,physical_stability_verified=False)


def mass_properties(vertices, faces, *, mass_kg=1., inertia_geometry='mesh'):
    import numpy as np
    import trimesh
    if isinstance(mass_kg,bool) or not np.isfinite(mass_kg) or mass_kg<=0:
        raise ValueError('Positive finite explicit object mass required')
    vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
        or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
        or not len(faces) or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Finite metric vertices and triangle indices required')
    if inertia_geometry not in ('mesh','convex_hull_if_open'):
        raise ValueError('Unknown inertia geometry mode')
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    source_watertight=bool(mesh.is_watertight and mesh.is_winding_consistent)
    proxy_used=False
    if not source_watertight:
        if inertia_geometry!='convex_hull_if_open':
            raise ValueError('Mesh-derived volume inertials require a closed consistently wound mesh')
        # The proxy changes inertial numbers only. Visual/collision geometry and
        # its link coordinates are kept byte-for-byte in write_urdf below.
        mesh=mesh.convex_hull
        proxy_used=True
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('A closed convex inertia proxy could not be built')
    if mesh.volume<0:mesh.invert()
    if not np.isfinite(mesh.volume) or mesh.volume<=1e-12:raise ValueError('Positive nondegenerate mesh volume required')
    center=np.asarray(mesh.center_mass);inertia=np.asarray(mesh.moment_inertia)*(float(mass_kg)/mesh.mass)
    eigen=np.linalg.eigvalsh(inertia)
    if not np.isfinite(center).all() or not np.isfinite(inertia).all() or eigen.min()<=0 or eigen.max()>eigen.sum()-eigen.max()+1e-10:
        raise ValueError('Physically valid positive rigid-body inertia required')
    return dict(mass_kg=float(mass_kg),center_of_mass_m=center.tolist(),inertia_kg_m2=inertia.tolist(),
        mesh_volume_m3=float(mesh.volume),assumption=('Uniform density over a closed convex proxy of the open reconstructed surface; supplied nominal mass' if proxy_used else 'Uniform density over the closed reconstructed volume; supplied nominal mass'),
        actual_mass_measured=False,geometry_changed=False,link_frame_changed=False,
        source_watertight=source_watertight,inertia_geometry='convex_hull_proxy' if proxy_used else 'source_mesh',
        inertia_proxy_approximate=proxy_used)


def write_urdf(mesh_path, output, *, mass_kg=1., inertia_geometry='mesh'):
    import json
    import numpy as np
    import trimesh
    from real2sim2real.artifacts import digest
    output=Path(output);source=(output.parent/mesh_path).resolve()
    mesh=trimesh.load(source,force='mesh',process=True)
    value=mass_properties(mesh.vertices,mesh.faces,mass_kg=mass_kg,inertia_geometry=inertia_geometry)
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
