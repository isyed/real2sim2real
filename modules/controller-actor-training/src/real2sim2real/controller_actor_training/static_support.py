"""Import existing support geometry as world shapes, without free support DOFs.

The temporary builder reuses the runtime's existing USD parser and material
defaults. Only its shapes are transferred; its kinematic body's free joint is
not added to the destination. Compatible with older pinned runtime snapshots.
"""
from dataclasses import dataclass,replace
import copy


@dataclass(frozen=True)
class StaticSupportBinding:
    shapes: object
    body_id: int = -1


def add_world_static_support(builder, path):
    import newton
    from flash_chord.scene.support import add_support_surfaces
    from flash_chord.scene.collision import ShapeSpan
    temporary=newton.ModelBuilder(up_axis=builder.up_axis,gravity=builder.gravity)
    temporary.default_shape_cfg=copy.copy(builder.default_shape_cfg)
    temporary.default_joint_cfg=copy.copy(builder.default_joint_cfg)
    newton.solvers.SolverMuJoCo.register_custom_attributes(temporary)
    source=add_support_surfaces(temporary,path)
    if source is None:return None
    before=(builder.body_count,builder.joint_count,builder.joint_coord_count,builder.joint_dof_count)
    start=builder.shape_count
    for i in source.shapes.ids():
        material={name:float(getattr(temporary,'shape_material_'+name)[i])
                  for name in ('ke','kd','kf','ka','mu','restitution')
                  if hasattr(temporary,'shape_material_'+name)}
        config=replace(builder.default_shape_cfg,density=0.,**material)
        kwargs=dict(body=-1,xform=temporary.shape_transform[i],cfg=config,label=temporary.shape_label[i])
        scale=temporary.shape_scale[i];kind=temporary.shape_type[i]
        if kind==newton.GeoType.MESH:
            builder.add_shape_mesh(mesh=temporary.shape_source[i],scale=scale,**kwargs)
        elif kind==newton.GeoType.BOX:
            builder.add_shape_box(hx=scale[0],hy=scale[1],hz=scale[2],**kwargs)
        elif kind==newton.GeoType.CYLINDER:
            builder.add_shape_cylinder(radius=scale[0],half_height=scale[2],**kwargs)
        else:
            raise ValueError('Unsupported static support shape')
    after=(builder.body_count,builder.joint_count,builder.joint_coord_count,builder.joint_dof_count)
    if before!=after:raise RuntimeError('Static supports introduced simulated bodies or joints')
    return StaticSupportBinding(ShapeSpan(start,builder.shape_count))
