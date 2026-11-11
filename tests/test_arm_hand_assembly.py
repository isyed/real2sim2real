import xml.etree.ElementTree as ET

import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_hand_assembly import (
    absolute_assets,
    build_assembly,
    extract_mounted_arm,
)
from real2sim2real.controller_actor_training.arm_path_audit import place_bases


def test_retained_fixture_copies_world_translation_axes():
    mujoco = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.arm_path_audit import (
        copy_fixture_frame,
    )

    xml = '<mujoco><worldbody><body name="hand"><joint name="slide" type="slide" axis="1 0 0"/><geom size=".02"/></body></worldbody></mujoco>'
    old = mujoco.MjModel.from_xml_string(xml)
    new = mujoco.MjModel.from_xml_string(xml)
    old.body_quat[old.body("hand").id] = [2**-0.5, 0, 0, 2**-0.5]
    old.jnt_axis[0] = [0, -1, 0]
    copy_fixture_frame(old, new, "hand")
    od, nd = mujoco.MjData(old), mujoco.MjData(new)
    od.qpos[0] = nd.qpos[0] = 0.3
    mujoco.mj_kinematics(old, od)
    mujoco.mj_kinematics(new, nd)
    np.testing.assert_allclose(nd.xpos[new.body("hand").id], [0.3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(nd.xpos, od.xpos)


def test_assets_get_names_before_namespace_and_degrees_rejected(tmp_path):
    source = tmp_path / "source.xml"
    mesh = tmp_path / "surface.stl"
    mesh.write_bytes(b"asset")
    source.write_text(
        '<mujoco><compiler angle="radian"/><asset><mesh file="surface.stl"/></asset></mujoco>'
    )
    root = ET.parse(source).getroot()
    hashes = absolute_assets(root, source)
    assert root.find("asset/mesh").get("name") == "surface"
    assert root.find("asset/mesh").get("file") == str(mesh)
    assert str(mesh) in hashes
    with pytest.raises(ValueError, match="radians"):
        absolute_assets(ET.fromstring("<mujoco/>"), source)


def test_extracted_mount_uses_mujoco_intrinsic_euler_convention(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    source = tmp_path / "robot.xml"
    hand = tmp_path / "hand.xml"
    source.write_text("""<mujoco><compiler angle="radian"/><asset/><worldbody>
    <body name="mount"><include file="hand.xml"/></body></worldbody>
    <actuator><motor joint="finger"/></actuator><keyframe><key/></keyframe></mujoco>""")
    hand.write_text("""<mujoco><body name="hand" pos=".1 .2 .3" euler=".6 -.9 1.2">
    <body><joint name="finger"/><geom size=".01"/></body></body></mujoco>""")
    receipt = extract_mounted_arm(source, hand, "hand", tmp_path / "arm.xml")
    expected = np.empty(4)
    mujoco.mju_euler2Quat(expected, np.array([0.6, -0.9, 1.2]), "xyz")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, expected)
    np.testing.assert_allclose(
        np.asarray(receipt["mount_from_hand"])[:3, :3], rotation.reshape(3, 3)
    )
    np.testing.assert_allclose(
        np.asarray(receipt["mount_from_hand"])[:3, 3], [0.1, 0.2, 0.3]
    )
    root = ET.parse(tmp_path / "arm.xml").getroot()
    assert not root.findall(".//include") and not root.findall("actuator/*")
    assert root.find("keyframe") is None


def test_two_independent_arms_preserve_hands_and_free_object(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    arm = tmp_path / "arm.xml"
    arm.write_text("""<mujoco><compiler angle="radian"/><default><geom friction=".1 .2 .3"/></default>
    <worldbody><body name="base"><body name="link"><joint name="elbow" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0 0 .3" size=".02"/><body name="mount" pos="0 0 .3"/>
    </body></body></worldbody></mujoco>""")
    scene = tmp_path / "scene.xml"
    scene.write_text("""<mujoco><compiler angle="radian"/><default><geom friction=".8 .005 .0001"/></default>
    <worldbody><body name="right"><joint name="fixture_r" type="slide"/><geom size=".02"/>
    <body name="right_tip" pos=".04 0 0"><joint name="finger_r" range="-1 1"/><geom size=".01"/></body></body>
    <body name="left"><joint name="fixture_l" type="slide"/><geom size=".02"/>
    <body name="left_tip" pos="-.04 0 0"><joint name="finger_l" range="-1 1"/><geom size=".01"/></body></body>
    <body name="object" pos="0 0 2"><freejoint name="object_free"/><geom size=".05"/></body></worldbody>
    <actuator><motor name="fixture_motor_r" joint="fixture_r"/><motor joint="finger_r"/>
    <motor name="fixture_motor_l" joint="fixture_l"/><motor joint="finger_l"/></actuator></mujoco>""")
    transform = np.eye(4)
    transform[2, 3] = 0.04
    spec = dict(
        arms=[
            dict(
                mjcf=str(arm),
                namespace=side + "__",
                hand_root=side,
                remove_fixture_joints=["fixture_" + suffix],
                mount_body="mount",
                mount_from_hand=transform.tolist(),
            )
            for side, suffix in [("right", "r"), ("left", "l")]
        ]
    )
    model, _ = build_assembly(scene, spec, tmp_path)
    assert model.njnt == 5 and model.nu == 2
    assert model.joint("object_free").type == mujoco.mjtJoint.mjJNT_FREE
    place_bases(
        model,
        {},
        dict(
            bases={
                "right__base": dict(position_m=[1, 0, 0], yaw_rad=0),
                "left__base": dict(position_m=[-1, 0, 0], yaw_rad=0),
            }
        ),
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body("right").id], [1, 0, 0.34])
    np.testing.assert_allclose(data.xpos[model.body("left").id], [-1, 0, 0.34])
    for name in ("right_tip", "left_tip"):
        body = model.body(name).id
        np.testing.assert_allclose(
            model.geom_friction[model.body_geomadr[body]], [0.8, 0.005, 0.0001]
        )
    assert model.body_parentid[model.body("right").id] == model.body("right__mount").id
    with pytest.raises(ValueError, match="top-level"):
        place_bases(
            model, {}, dict(bases={"right": dict(position_m=[0, 0, 0], yaw_rad=0)})
        )
    spec["arms"][0]["remove_fixture_joints"] = ["finger_r"]
    with pytest.raises(ValueError, match="fixture"):
        build_assembly(scene, spec, tmp_path)
