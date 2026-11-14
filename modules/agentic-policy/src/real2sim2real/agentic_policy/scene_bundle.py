"""Relocate a flattened MJCF's assets without changing geometry or physics."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from real2sim2real.artifacts import digest, write_json


def portable_scene(source, output, *, mapping=None):
    from real2sim2real.mujoco_assets import relocated_path

    source, output = Path(source).resolve(), Path(output).resolve()
    tree = ET.parse(source)
    if tree.findall(".//include"):
        raise ValueError("Flatten MJCF includes before exporting this scene bundle")
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    compiler = tree.find("compiler")
    roots = {
        k: (compiler.get(k, "") if compiler is not None else "")
        for k in ["meshdir", "texturedir", "assetdir"]
    }
    pins = {}
    for element in tree.iter():
        filename = element.get("file")
        if filename is None:
            continue
        if element.tag not in {"mesh", "texture", "hfield"}:
            raise ValueError("Unsupported external MJCF file type: " + element.tag)
        directory = (
            roots["meshdir" if element.tag == "mesh" else "texturedir"]
            or roots["assetdir"]
        )
        path = (
            Path(filename)
            if Path(filename).is_absolute()
            else source.parent / directory / filename
        )
        path = path.resolve()
        path = Path(relocated_path(str(path), mapping or {}))
        sha = digest(path)
        dest = assets / (sha + path.suffix)
        if not dest.exists():
            shutil.copyfile(path, dest)
        element.set("file", str(dest.relative_to(output)))
        pins[str(path)] = {"sha256": sha, "relative": str(dest.relative_to(output))}
    if compiler is not None:
        for key in roots:
            compiler.attrib.pop(key, None)
    target = output / "scene.xml"
    tree.write(target, encoding="unicode")
    write_json(
        output / "asset-provenance.json",
        {
            "source_sha256": digest(source),
            "assets": pins,
            "changes": "asset paths only; compiled model parity must be checked separately",
        },
    )
    return target


def scene_input_fingerprint(source):
    """Hash flattened MJCF semantics and referenced asset bytes, ignoring relocation.

    Includes collision/solver options, defaults and all model elements. This is
    input parity, not a guarantee about compiled models or runtime overrides.
    """
    import hashlib
    import json

    source = Path(source).resolve()
    tree = ET.parse(source)
    if tree.findall(".//include"):
        raise ValueError("Flatten MJCF includes before checking input parity")
    compiler = tree.find("compiler")
    roots = {
        k: compiler.get(k, "") if compiler is not None else ""
        for k in ["meshdir", "texturedir", "assetdir"]
    }

    def encode(element):
        attributes = dict(element.attrib)
        filename = attributes.get("file")
        if filename is not None:
            if element.tag not in {"mesh", "texture", "hfield"}:
                raise ValueError("Unsupported external MJCF file type: " + element.tag)
            key = {"mesh": "meshdir", "texture": "texturedir"}.get(
                element.tag, "assetdir"
            )
            directory = roots[key] or roots["assetdir"]
            asset = Path(filename)
            if not asset.is_absolute():
                asset = source.parent / directory / asset
            attributes["file"] = "sha256:" + digest(asset.resolve())
        if element.tag == "compiler":
            for key in roots:
                attributes.pop(key, None)
        return [
            element.tag,
            attributes,
            (element.text or "").strip(),
            [encode(child) for child in element],
        ]

    return hashlib.sha256(
        json.dumps(encode(tree.getroot()), sort_keys=True).encode()
    ).hexdigest()


def require_scene_input_parity(host_scene, author_scene):
    """Reject stale author scenes even when meshes match but collision flags differ."""
    host = scene_input_fingerprint(host_scene)
    author = scene_input_fingerprint(author_scene)
    if host != author:
        raise ValueError(
            "Author/host scene inputs differ; verify collision options, geometry and assets before authoring"
        )
    return {
        "scene_input_fingerprint": host,
        "host_xml_sha256": digest(Path(host_scene)),
        "author_xml_sha256": digest(Path(author_scene)),
        "input_parity": True,
        "scope": "Flattened MJCF and asset bytes; runtime code/state parity is separate",
    }
