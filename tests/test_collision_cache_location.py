from flash_chord.assets import mesh


def test_active_assets_can_change_after_mesh_module_import(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASH_CHORD_CACHE_DIR", raising=False)
    monkeypatch.setenv("R2S2R_ASSETS", str(tmp_path / "policy_a"))
    assert mesh.default_cache_dir() == tmp_path / "policy_a/.cache/convex"
    monkeypatch.setenv("R2S2R_ASSETS", str(tmp_path / "policy_b"))
    assert mesh.default_cache_dir() == tmp_path / "policy_b/.cache/convex"


def test_explicit_cache_override_has_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("R2S2R_ASSETS", str(tmp_path / "policy"))
    monkeypatch.setenv("FLASH_CHORD_CACHE_DIR", str(tmp_path / "cache"))
    assert mesh.default_cache_dir() == tmp_path / "cache/convex"


def test_default_assets_remain_supported(monkeypatch):
    monkeypatch.delenv("R2S2R_ASSETS", raising=False)
    monkeypatch.delenv("FLASH_CHORD_CACHE_DIR", raising=False)
    assert mesh.default_cache_dir() == mesh.ASSETS_DIR / ".cache/convex"
