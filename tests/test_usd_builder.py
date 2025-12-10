import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from recipe_compiler.usd_builder import StubLayer, USDSceneBuilder  # noqa: E402


def _load_catalog_sample() -> tuple[str, str]:
    catalog_path = PROJECT_ROOT / "asset_index.json"
    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)

    sample_asset = catalog["assets"][0]["relative_path"]
    pack_name = catalog["pack_info"]["name"]
    return pack_name, sample_asset


def test_resolve_asset_path_wraps_asset_reference_in_stub_mode():
    pack_name, asset_path = _load_catalog_sample()
    builder = USDSceneBuilder()

    asset_root = "/mnt/assets"
    resolved = builder._resolve_asset_path(asset_path, asset_root)

    expected = str(Path(asset_root) / pack_name / asset_path)
    assert resolved == f"@{expected}@"


def test_validate_catalog_path_accepts_indexed_asset():
    _, asset_path = _load_catalog_sample()
    builder = USDSceneBuilder()

    # Should not raise for a valid path
    builder._validate_catalog_path(asset_path)


def test_validate_catalog_path_rejects_unknown_asset():
    builder = USDSceneBuilder()

    with pytest.raises(AssertionError):
        builder._validate_catalog_path("nonexistent/usd_asset.usd")


def test_build_layout_enforces_catalog_membership_before_reference():
    builder = USDSceneBuilder()
    layout_layer = StubLayer("layout", "/tmp/layout.usda")

    objects = [
        {
            "id": "obj_missing",
            "transform": {},
            "chosen_asset": {"asset_path": "not/in/catalog.usd"},
        }
    ]

    with pytest.raises(AssertionError):
        builder.build_layout(layout_layer, objects, "/assets")


def test_build_layout_accepts_catalog_asset_in_stub_mode():
    _, asset_path = _load_catalog_sample()
    builder = USDSceneBuilder()
    layout_layer = StubLayer("layout", "/tmp/layout.usda")

    objects = [
        {
            "id": "obj_valid",
            "transform": {},
            "chosen_asset": {"asset_path": asset_path},
        }
    ]

    builder.build_layout(layout_layer, objects, "/assets")

    assert "/Objects/obj_valid" in layout_layer.prims
    assert layout_layer.prims["/Objects/obj_valid"]["data"].get("asset_ref") == asset_path
