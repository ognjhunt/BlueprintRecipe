import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from asset_catalog.asset_matcher import AssetMatcher  # noqa: E402
from asset_catalog.catalog_builder import AssetCatalog, AssetEntry  # noqa: E402


def _make_catalog(assets: list[AssetEntry]) -> AssetCatalog:
    categories = sorted({asset.category for asset in assets})
    return AssetCatalog(
        pack_name="test_pack",
        display_name="Test Pack",
        version_hash="hash",
        indexed_at=datetime.utcnow().isoformat(),
        total_assets=len(assets),
        categories=categories,
        assets=assets,
    )


def test_filter_excludes_oversized_assets():
    assets = [
        AssetEntry(
            asset_id="chair_small",
            relative_path="chair_small.usd",
            file_type=".usd",
            category="chair",
            tags=["chair"],
            dimensions={"width": 1.0, "depth": 1.0, "height": 1.0},
        ),
        AssetEntry(
            asset_id="chair_large",
            relative_path="chair_large.usd",
            file_type=".usd",
            category="chair",
            tags=["chair"],
            dimensions={"width": 10.0, "depth": 10.0, "height": 10.0},
        ),
    ]

    matcher = AssetMatcher(_make_catalog(assets))

    result = matcher.match(
        {
            "id": "obj1",
            "category": "chair",
            "description": "wood chair",
            "estimated_dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0},
        },
        dimension_filter_ratio=3.0,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].asset_id == "chair_small"


def test_filter_excludes_undersized_assets():
    assets = [
        AssetEntry(
            asset_id="table_tiny",
            relative_path="table_tiny.usd",
            file_type=".usd",
            category="table",
            tags=["table"],
            dimensions={"width": 0.1, "depth": 0.1, "height": 0.1},
        ),
        AssetEntry(
            asset_id="table_normal",
            relative_path="table_normal.usd",
            file_type=".usd",
            category="table",
            tags=["table"],
            dimensions={"width": 1.0, "depth": 1.0, "height": 1.0},
        ),
    ]

    matcher = AssetMatcher(_make_catalog(assets))

    result = matcher.match(
        {
            "id": "obj2",
            "category": "table",
            "description": "small table",
            "estimated_dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0},
        },
        dimension_filter_ratio=3.0,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].asset_id == "table_normal"


def test_logs_when_all_candidates_filtered(caplog: pytest.LogCaptureFixture):
    assets = [
        AssetEntry(
            asset_id="sofa_large",
            relative_path="sofa_large.usd",
            file_type=".usd",
            category="sofa",
            tags=["sofa"],
            dimensions={"width": 20.0, "depth": 20.0, "height": 10.0},
        ),
    ]

    matcher = AssetMatcher(_make_catalog(assets))

    with caplog.at_level(logging.INFO):
        result = matcher.match(
            {
                "id": "obj3",
                "category": "sofa",
                "description": "sofa",
                "estimated_dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0},
            },
            dimension_filter_ratio=2.0,
        )

    assert not result.candidates
    assert any(
        "All candidates filtered out due to dimension ratio" in record.message
        for record in caplog.records
    )


def test_low_confidence_auto_select_logs(caplog: pytest.LogCaptureFixture):
    assets = [
        AssetEntry(
            asset_id="chair_basic",
            relative_path="chair_basic.usd",
            file_type=".usd",
            category="chair",
            tags=["chair"],
        ),
    ]

    matcher = AssetMatcher(_make_catalog(assets))

    with caplog.at_level(logging.INFO):
        result = matcher.match(
            {
                "id": "obj4",
                "category": "chair",
                "description": "chair with cushion",
            }
        )

    assert result.chosen is not None
    assert any(
        "Auto-selected low-confidence candidate" in warning
        for warning in result.warnings
    )
    assert any(
        "Auto-selected low-confidence candidate for object" in record.message
        for record in caplog.records
    )


def test_auto_select_respects_floor_threshold():
    assets = [
        AssetEntry(
            asset_id="lamp_basic",
            relative_path="lamp_basic.usd",
            file_type=".usd",
            category="lamp",
            tags=["lamp"],
        ),
    ]

    matcher = AssetMatcher(
        _make_catalog(assets),
        auto_select_floor=0.6,
        auto_select_ceiling=0.8,
    )

    result = matcher.match(
        {
            "id": "obj5",
            "category": "lamp",
            "description": "lamp",
        }
    )

    assert result.chosen is None
    assert any(candidate.asset_id == "lamp_basic" for candidate in result.candidates)
