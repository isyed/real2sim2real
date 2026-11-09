import pytest
pytest.importorskip("newton", reason="Requires the optional simulation runtime")
pytest.importorskip("pxr", reason="Requires the optional USD runtime")


def support_file(tmp_path):
    path=tmp_path/'support.usda'
    path.write_text('''#usda 1.0
(metersPerUnit = 1
 upAxis = "Z")
def Cube "table" {
 double size = 1
 double3 xformOp:scale = (1, 1, 0.1)
 double3 xformOp:translate = (0, 0, 0.8)
 uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
 bool physics:collisionEnabled = true
}
''')
    return path


@pytest.mark.parametrize('static_world', [False,True])
def test_support_representation_has_exact_declared_topology(tmp_path,monkeypatch,static_world):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    import newton,warp as wp
    from flash_chord.scene.support import add_support_surfaces
    with wp.ScopedDevice('cpu'):
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        binding=add_support_surfaces(builder,support_file(tmp_path),static_world=static_world)
    assert builder.shape_count==1
    assert builder.body_count==(0 if static_world else 1)
    assert builder.joint_coord_count==(0 if static_world else 7)
    assert binding.body_id==(-1 if static_world else 0)


def test_cpu_isolated_test_keeps_support_out_of_free_body_dynamics(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('FLASH_CHORD_CACHE_DIR',str(tmp_path/'cache'))
    import warp as wp
    monkeypatch.setattr(wp.config,'kernel_cache_dir',str(tmp_path/'warp'))
    import trimesh
    from real2sim2real.controller_actor_training.object_stability import run,assess_settling
    mesh=tmp_path/'cube.obj';trimesh.creation.box(extents=[.1,.1,.1]).export(mesh)
    urdf=tmp_path/'cube.urdf'
    urdf.write_text('''<robot name="test"><link name="object">
<inertial><mass value="1"/><inertia ixx=".001667" iyy=".001667" izz=".001667" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><geometry><mesh filename="cube.obj"/></geometry></visual>
<collision><geometry><mesh filename="cube.obj"/></geometry></collision>
</link></robot>''')
    value=run(urdf,[0,0,.91,0,0,0,1],tmp_path/'result',seconds=2.,device='cpu',support_usda=support_file(tmp_path))
    assert value['simulated_body_count']==1 and value['simulated_joint_coordinate_count']==7
    assert value['support_motion']['fixed_by_construction']
    assert value['support_shape_count']==1 and value['support_mode']=='world_static'
    assert value['finite'] and value['max_displacement_m']<.05
    # CPU single-contact box rocking is not a universal settling guarantee.
    assert not assess_settling(value)['training_ready_established']
