import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.scene_planner import ScenePlanner


def _base_scene_plan():
    return {
        "environment_analysis": {},
        "object_inventory": [],
        "spatial_layout": {},
    }


def test_validate_dimensions_clamps_tall_fridge():
    planner = ScenePlanner()
    scene_plan = _base_scene_plan()
    scene_plan["object_inventory"].append(
        {
            "id": "fridge1",
            "category": "refrigerator",
            "dimensions": {"height": 2.5, "width": 0.8, "depth": 0.7},
        }
    )

    warnings = planner.validate_plan(scene_plan)

    assert any("fridge1 refrigerator height" in warning for warning in warnings)
    assert scene_plan["object_inventory"][0]["dimensions"]["height"] == 2.2
    assert "validation_warnings" in scene_plan


def test_validate_dimensions_flags_small_mug():
    planner = ScenePlanner()
    scene_plan = _base_scene_plan()
    scene_plan["object_inventory"].append(
        {
            "id": "mug1",
            "category": "mug",
            "dimensions": {"height": 0.05, "width": 0.08, "depth": 0.08},
        }
    )

    warnings = planner.validate_plan(scene_plan)

    assert any("mug1 mug height" in warning for warning in warnings)
    assert scene_plan["object_inventory"][0]["dimensions"]["height"] == 0.07

