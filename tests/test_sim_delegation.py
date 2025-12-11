import json
from pathlib import Path

import pytest

from src.recipe_compiler import RecipeCompiler
from src.recipe_compiler.compiler import CompilerConfig
from src.sim_integration.blueprint_sim import BlueprintSimClient


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def generate_from_manifest(self, manifest, output_dir, policies=None, metadata=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scene.usda").write_text("delegate")
        (output_dir / "scene_manifest.json").write_text(json.dumps(manifest))
        (output_dir / "recipe.json").write_text(json.dumps({"version": "1.0.0"}))
        (output_dir / "bundle.zip").write_text("bundle")
        (output_dir / "isaac_lab.tar").write_text("lab")

        self.calls.append(
            {
                "manifest": manifest,
                "output_dir": str(output_dir),
                "policies": policies or [],
                "metadata": metadata or {},
            }
        )

        return {
            "scene_usd": str(output_dir / "scene.usda"),
            "manifest_path": str(output_dir / "scene_manifest.json"),
            "recipe_path": str(output_dir / "recipe.json"),
            "layer_paths": {"layout": str(output_dir / "layout.usda")},
            "replicator_bundle": str(output_dir / "bundle.zip"),
            "isaac_lab_bundle": str(output_dir / "isaac_lab.tar"),
            "qa_report": {"checks": {"delegated": True}},
        }


@pytest.fixture()
def sample_manifest():
    return {
        "version": "1.0.0",
        "scene": {"meters_per_unit": 1.0},
        "assets": {"asset_root": "/assets", "packs": []},
        "objects": [],
    }


def test_client_normalizes_pipeline_result(tmp_path, sample_manifest):
    pipeline = _FakePipeline()
    client = BlueprintSimClient(pipeline=pipeline)

    result = client.generate_from_manifest(sample_manifest, output_dir=tmp_path)

    assert result.scene_usd_path.endswith("scene.usda")
    assert result.replicator_bundle.endswith("bundle.zip")
    assert pipeline.calls and pipeline.calls[0]["manifest"]["version"] == "1.0.0"


def test_recipe_compiler_delegates_to_pipeline(tmp_path):
    pipeline = _FakePipeline()
    sim_client = BlueprintSimClient(pipeline=pipeline)

    compiler = RecipeCompiler(
        CompilerConfig(output_dir=str(tmp_path), asset_root="/assets"),
        sim_client=sim_client,
    )

    scene_plan = {
        "version": "1.0.0",
        "environment_analysis": {"detected_type": "kitchen", "estimated_dimensions": {}},
        "object_inventory": [
            {
                "id": "obj-1",
                "category": "mug",
                "estimated_dimensions": {"width": 0.1, "depth": 0.1, "height": 0.1},
            }
        ],
        "spatial_layout": {"placements": [{"object_id": "obj-1", "position": {"x": 0, "y": 0, "z": 0}}]},
    }
    matched_assets = {"obj-1": {"chosen_path": "pack/mug.usd", "pack_name": "pack"}}

    result = compiler.compile(scene_plan, matched_assets, metadata={"recipe_id": "scene-123"})

    assert result.success
    assert result.scene_path.endswith("scene.usda")
    assert result.replicator_bundle and result.replicator_bundle.endswith("bundle.zip")
    assert pipeline.calls, "Blueprint Sim pipeline was not invoked"
    assert pipeline.calls[0]["manifest"]["objects"]
