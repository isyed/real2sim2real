"""Load archived MJCF/URDF with explicit asset-root relocation, on CPU.

Original scene files remain immutable. Only file-path attributes are remapped.
With no mapping configured, loading delegates directly to MuJoCo. Temporary
siblings preserve each document's relative-path base and are always removed.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

PATH_ATTRIBUTES = {'file', 'filename', 'meshdir', 'texturedir', 'assetdir'}


def path_mapping():
    config = os.environ.get('R2S2R_ASSET_PATH_MAP')
    mapping = json.loads(Path(config).read_text()) if config else {}
    if not isinstance(mapping, dict) or any(
        not isinstance(a, str) or not isinstance(b, str) or
        not Path(a).is_absolute() or not Path(b).is_absolute()
        for a, b in mapping.items()
    ):
        raise ValueError('Asset relocation requires absolute source/destination roots')
    return mapping


def relocated_path(value, mapping):
    for old in sorted(mapping, key=len, reverse=True):
        if value == old or value.startswith(old.rstrip('/') + '/'):
            return mapping[old].rstrip('/') + value[len(old.rstrip('/')):]
    return value


@contextmanager
def relocated_scene(source, mapping):
    source = Path(relocated_path(str(source), mapping)).resolve()
    temporary = []
    active = set()

    def stage(path):
        path = path.resolve()
        if path in active:
            raise ValueError('Cyclic XML include during asset relocation')
        active.add(path)
        tree = ET.parse(path)
        changed = False
        for element in tree.iter():
            for key, original in list(element.attrib.items()):
                if key not in PATH_ATTRIBUTES:
                    continue
                value = relocated_path(original, mapping)
                if element.tag == 'include' and key == 'file':
                    nested = Path(value)
                    if not nested.is_absolute():
                        nested = path.parent / nested
                    mapped = stage(nested)
                    if mapped != nested.resolve():
                        value = str(mapped)
                if value != original:
                    element.set(key, value)
                    changed = True
        active.remove(path)
        if not changed:
            return path
        with tempfile.NamedTemporaryFile(prefix='.r2s2r-assets-', suffix=path.suffix,
                                         dir=path.parent, delete=False) as stream:
            target = Path(stream.name)
            temporary.append(target)
            tree.write(stream, encoding='utf-8', xml_declaration=True)
        return target

    try:
        yield stage(source)
    finally:
        for path in reversed(temporary):
            path.unlink(missing_ok=True)


def load_model(source, *, mapping=None, **kwargs):
    import mujoco
    mapping = path_mapping() if mapping is None else mapping
    if not mapping:
        return mujoco.MjModel.from_xml_path(str(source), **kwargs)
    with relocated_scene(source, mapping) as path:
        return mujoco.MjModel.from_xml_path(str(path), **kwargs)
