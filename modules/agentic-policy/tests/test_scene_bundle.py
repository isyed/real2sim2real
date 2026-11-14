import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from real2sim2real.agentic_policy.scene_bundle import portable_scene


def test_relocated_mesh_preserves_compiled_physics(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetra.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    source = tmp_path / "original.xml"
    source.write_text(
        '<mujoco><compiler meshdir="meshes"/><asset><mesh name="part" file="tetra.obj"/></asset><worldbody><body><geom type="mesh" mesh="part"/></body></worldbody></mujoco>'
    )
    original = source.read_bytes()
    relocated = portable_scene(source, tmp_path / "export")
    a = mujoco.MjModel.from_xml_path(str(source))
    b = mujoco.MjModel.from_xml_path(str(relocated))
    for name in [
        "mesh_vert",
        "mesh_face",
        "geom_size",
        "geom_pos",
        "body_mass",
        "body_inertia",
    ]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert source.read_bytes() == original
    assert all(
        not Path(e.get("file")).is_absolute()
        for e in ET.parse(relocated).iter()
        if e.get("file")
    )


def test_includes_are_not_silently_dropped(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text('<mujoco><include file="other.xml"/></mujoco>')
    with pytest.raises(ValueError, match="Flatten"):
        portable_scene(source, tmp_path / "output")


def test_author_scene_flags_cannot_disagree_with_host(tmp_path):
    from real2sim2real.agentic_policy.scene_bundle import require_scene_input_parity

    host, author = tmp_path / "host.xml", tmp_path / "author.xml"
    host.write_text(
        '<mujoco><option><flag multiccd="disable"/></option><worldbody/></mujoco>'
    )
    author.write_text("<mujoco><option/><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="scene inputs differ"):
        require_scene_input_parity(host, author)
    author.write_bytes(host.read_bytes())
    assert require_scene_input_parity(host, author)["input_parity"]


def test_parity_checks_assets_but_allows_relocation(tmp_path):
    from real2sim2real.agentic_policy.scene_bundle import require_scene_input_parity

    source = tmp_path / "source.xml"
    (tmp_path / "mesh.obj").write_text("asset bytes")
    source.write_text(
        '<mujoco><asset><mesh name="m" file="mesh.obj"/></asset><worldbody/></mujoco>'
    )
    target = portable_scene(source, tmp_path / "export")
    assert require_scene_input_parity(source, target)["input_parity"]
    asset = next((tmp_path / "export/assets").iterdir())
    asset.write_text("different geometry bytes")
    with pytest.raises(ValueError, match="scene inputs differ"):
        require_scene_input_parity(source, target)


def test_campaign_rejects_declared_scene_mismatch_before_freezing(tmp_path):
    from real2sim2real.agentic_policy import campaign

    from real2sim2real.artifacts import write_json

    bundle = tmp_path / "bundle"
    (bundle / "author").mkdir(parents=True)
    (bundle / "scene.xml").write_text(
        '<mujoco><option><flag multiccd="disable"/></option><worldbody/></mujoco>'
    )
    (bundle / "author/scene.xml").write_text("<mujoco><worldbody/></mujoco>")
    write_json(bundle / "runtime.json", {"scene": "scene.xml"})
    write_json(
        bundle / "experiment.json",
        {
            "mode": "manual",
            "feedback_revisions": 0,
            "rollout_timeout_s": 1,
            "backend": {"kind": "continuous", "runtime": "runtime.json"},
            "author_scene": "author/scene.xml",
        },
    )
    with pytest.raises(ValueError, match="scene inputs differ"):
        campaign.initialize(bundle, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()
