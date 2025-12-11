import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from manifest_adapter import SceneManifestAdapter  # noqa: E402
from utils.schema_validator import validate_manifest  # noqa: E402


@pytest.fixture
def sample_scene_plan() -> dict:
    return {
        "version": "0.1.0",
        "environment_analysis": {
            "detected_type": "kitchen",
            "estimated_dimensions": {"width": 4.0, "depth": 3.5, "height": 2.5},
        },
        "object_inventory": [
            {
                "id": "obj_1",
                "category": "cabinet",
                "description": "Lower kitchen cabinet",
                "is_articulated": True,
                "articulation_type": "door",
                "estimated_dimensions": {"width": 0.8, "depth": 0.5, "height": 0.9},
            },
            {
                "id": "obj_2",
                "category": "bottle",
                "description": "Water bottle on counter",
                "is_manipulable": True,
                "is_variation_candidate": True,
            },
        ],
        "spatial_layout": {
            "placements": [
                {
                    "object_id": "obj_1",
                    "position": {"x": 1.0, "y": 0.0, "z": 0.5, "reference": "room_center"},
                    "rotation_degrees": 90,
                },
                {
                    "object_id": "obj_2",
                    "position": {"x": 1.2, "y": 0.9, "z": 0.6},
                    "rotation_degrees": 0,
                },
            ],
            "relationships": [
                {"type": "on_top_of", "subject_id": "obj_2", "object_id": "obj_1"}
            ],
            "functional_zones": [
                {"name": "countertop", "object_ids": ["obj_2"], "type": "surface"}
            ],
        },
    }


@pytest.fixture
def matched_assets() -> dict:
    return {
        "obj_1": {
            "chosen_path": "pack/cabinet.usd",
            "variants": {"color": "white"},
            "pack_name": "kitchen_pack",
            "simready_metadata": {"is_simready": True},
        },
        "obj_2": {
            "chosen_path": "pack/bottle.usd",
            "pack_name": "kitchen_pack",
            "candidates": [
                {"asset_path": "pack/bottle.usd", "score": 0.9, "dimensions": {"width": 0.08, "depth": 0.08, "height": 0.25}}
            ],
        },
    }


@pytest.fixture
def recipe(sample_scene_plan: dict) -> dict:
    return {
        "objects": [
            {
                "id": "obj_1",
                "logical_type": "cabinet",
                "category": "cabinet",
                "transform": {
                    "position": {"x": 1.0, "y": 0.0, "z": 0.5},
                    "rotation": {"w": 0.707, "x": 0, "y": 0.707, "z": 0},
                    "scale": {"x": 1, "y": 1, "z": 1},
                },
                "chosen_asset": {"asset_path": "pack/cabinet.usd", "variant_selections": {"color": "white"}},
                "semantics": {"class": "cabinet", "instance_id": "obj_1"},
                "physics": {"enabled": True, "rigid_body": False},
                "articulation": {"type": "revolute", "axis": "y", "limits": {"lower": 0, "upper": 1.5}},
            },
            {
                "id": "obj_2",
                "logical_type": "bottle",
                "category": "bottle",
                "transform": {
                    "position": {"x": 1.2, "y": 0.9, "z": 0.6},
                    "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                    "scale": {"x": 1, "y": 1, "z": 1},
                },
                "chosen_asset": {"asset_path": "pack/bottle.usd", "variant_selections": {}},
                "semantics": {"class": "bottle", "instance_id": "obj_2"},
                "physics": {"enabled": True, "rigid_body": True},
            },
        ]
    }


def test_manifest_adapter_builds_expected_manifest(sample_scene_plan, matched_assets, recipe):
    adapter = SceneManifestAdapter(
        asset_root="/assets",
        meters_per_unit=1.0,
        coordinate_frame="y_up",
        up_axis="Y",
    )

    manifest = adapter.build_manifest(sample_scene_plan, matched_assets, recipe, metadata={"recipe_id": "scene123"})

    assert manifest["scene"]["environment_type"] == "kitchen"
    assert manifest["assets"]["asset_root"] == "/assets"
    assert {obj["id"] for obj in manifest["objects"]} == {"obj_1", "obj_2"}

    cabinet = next(obj for obj in manifest["objects"] if obj["id"] == "obj_1")
    assert cabinet["sim_role"] == "articulated"
    assert cabinet["asset"]["path"] == "pack/cabinet.usd"
    assert cabinet["articulation"]["axis"] == "y"

    bottle = next(obj for obj in manifest["objects"] if obj["id"] == "obj_2")
    assert bottle["sim_role"] == "interactive"
    assert bottle["placement_region"] == "countertop"
    assert bottle["physics_hints"]["estimated_dimensions"]["height"] == 0.25

    validation = validate_manifest(manifest)
    assert validation["valid"] is True


def test_manifest_validation_flags_missing_asset(sample_scene_plan, matched_assets, recipe):
    adapter = SceneManifestAdapter(asset_root="/assets", meters_per_unit=1.0)
    manifest = adapter.build_manifest(sample_scene_plan, matched_assets, recipe)

    # Corrupt manifest: remove a required asset path
    manifest["objects"][0]["asset"]["path"] = ""

    validation = validate_manifest(manifest)
    assert validation["valid"] is False
    assert any("asset/path" in err or "asset/path".replace("/", "") in err for err in validation["errors"])

