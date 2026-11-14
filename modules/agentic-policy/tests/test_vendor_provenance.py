"""Executable bundled sources must match their recorded integration patches."""

import hashlib
import json
from pathlib import Path
from typing import Any, get_type_hints

from real2sim2real.agentic_policy import _vendor
from real2sim2real.agentic_policy._vendor.geometry.tools import AttachmentResult


def test_bundled_component_hashes_match_provenance():
    root = Path(_vendor.__file__).parent
    manifest = json.loads((root / "provenance.json").read_text())
    for component, record in manifest.items():
        for name, hashes in record["files"].items():
            assert hashlib.sha256((root / component / name).read_bytes()).hexdigest() == hashes["bundled_sha256"], (component, name)


def test_attachment_result_annotations_resolve_for_tool_schemas():
    assert get_type_hints(AttachmentResult) == {"attached_object": dict[str, Any]}
