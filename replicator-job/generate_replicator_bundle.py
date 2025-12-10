#!/usr/bin/env python3
"""
Replicator Bundle Generator for Isaac Sim Synthetic Data Generation.

This script analyzes a completed scene and generates:
1. Placement regions as USD layers (sink_region, counter_region, etc.)
2. Policy-specific Replicator Python scripts
3. Variation asset manifests (dirty dishes, groceries, clothes, etc.)
4. Configuration files for different training policies

The output bundle is ready to be loaded into Isaac Sim for domain randomization
and synthetic data generation.
"""

from __future__ import annotations

import json
import os
import sys
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    types = None

GCS_ROOT = Path("/mnt/gcs")


class EnvironmentType(str, Enum):
    KITCHEN = "kitchen"
    GROCERY = "grocery"
    WAREHOUSE = "warehouse"
    LOADING_DOCK = "loading_dock"
    LAB = "lab"
    OFFICE = "office"
    UTILITY_ROOM = "utility_room"
    HOME_LAUNDRY = "home_laundry"
    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"
    BATHROOM = "bathroom"
    GENERIC = "generic"


class PolicyTarget(str, Enum):
    DEXTEROUS_PICK_PLACE = "dexterous_pick_place"
    ARTICULATED_ACCESS = "articulated_access"
    PANEL_INTERACTION = "panel_interaction"
    MIXED_SKU_LOGISTICS = "mixed_sku_logistics"
    PRECISION_INSERTION = "precision_insertion"
    LAUNDRY_SORTING = "laundry_sorting"
    DISH_LOADING = "dish_loading"
    GROCERY_STOCKING = "grocery_stocking"
    TABLE_CLEARING = "table_clearing"
    DRAWER_MANIPULATION = "drawer_manipulation"
    DOOR_MANIPULATION = "door_manipulation"
    KNOB_MANIPULATION = "knob_manipulation"
    GENERAL_MANIPULATION = "general_manipulation"


SCENE_TYPE_TO_ENVIRONMENT = {
    "kitchen": EnvironmentType.KITCHEN,
    "grocery": EnvironmentType.GROCERY,
    "warehouse": EnvironmentType.WAREHOUSE,
    "loading_dock": EnvironmentType.LOADING_DOCK,
    "lab": EnvironmentType.LAB,
    "office": EnvironmentType.OFFICE,
    "utility_room": EnvironmentType.UTILITY_ROOM,
    "laundry": EnvironmentType.HOME_LAUNDRY,
    "laundry_room": EnvironmentType.HOME_LAUNDRY,
    "bedroom": EnvironmentType.BEDROOM,
    "living_room": EnvironmentType.LIVING_ROOM,
    "bathroom": EnvironmentType.BATHROOM,
}

ENVIRONMENT_DEFAULT_POLICIES = {
    EnvironmentType.KITCHEN: [
        PolicyTarget.DISH_LOADING,
        PolicyTarget.TABLE_CLEARING,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
    ],
    EnvironmentType.GROCERY: [
        PolicyTarget.GROCERY_STOCKING,
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
    ],
    EnvironmentType.WAREHOUSE: [
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.LOADING_DOCK: [
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.LAB: [
        PolicyTarget.PRECISION_INSERTION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
        PolicyTarget.DRAWER_MANIPULATION,
    ],
    EnvironmentType.OFFICE: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.PANEL_INTERACTION,
    ],
    EnvironmentType.UTILITY_ROOM: [
        PolicyTarget.PANEL_INTERACTION,
        PolicyTarget.KNOB_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.ARTICULATED_ACCESS,
    ],
    EnvironmentType.HOME_LAUNDRY: [
        PolicyTarget.LAUNDRY_SORTING,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.KNOB_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
    ],
    EnvironmentType.BEDROOM: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.LAUNDRY_SORTING,
    ],
    EnvironmentType.LIVING_ROOM: [
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.BATHROOM: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
    ],
    EnvironmentType.GENERIC: [
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
}


@dataclass
class PlacementRegion:
    name: str
    description: str
    surface_type: str
    parent_object_id: Optional[str] = None
    position: Optional[List[float]] = None
    size: Optional[List[float]] = None
    rotation: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)
    suitable_for: List[str] = field(default_factory=list)


@dataclass
class VariationAsset:
    name: str
    category: str
    description: str
    semantic_class: str
    priority: str
    source_hint: Optional[str] = None
    example_variants: List[str] = field(default_factory=list)
    physics_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomizerConfig:
    name: str
    enabled: bool = True
    frequency: str = "per_frame"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    policy_id: str
    policy_name: str
    policy_target: PolicyTarget
    description: str
    placement_regions: List[str] = field(default_factory=list)
    variation_assets: List[str] = field(default_factory=list)
    randomizers: List[RandomizerConfig] = field(default_factory=list)
    capture_config: Dict[str, Any] = field(default_factory=dict)
    scene_modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicatorBundle:
    scene_id: str
    environment_type: EnvironmentType
    scene_type: str
    policies: List[PolicyConfig] = field(default_factory=list)
    global_placement_regions: List[PlacementRegion] = field(default_factory=list)
    global_variation_assets: List[VariationAsset] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_gemini_client():
    if genai is None:
        raise ImportError("google-genai package not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def detect_environment_type(scene_type: str, inventory: dict) -> EnvironmentType:
    scene_type_lower = scene_type.lower().strip()
    if scene_type_lower in SCENE_TYPE_TO_ENVIRONMENT:
        return SCENE_TYPE_TO_ENVIRONMENT[scene_type_lower]

    objects = inventory.get("objects", [])
    object_ids = {obj.get("id", "").lower() for obj in objects}

    if any(keyword in oid for keyword in ["refrigerator", "oven", "dishwasher"] for oid in object_ids):
        return EnvironmentType.KITCHEN
    if any(keyword in oid for keyword in ["washer", "dryer", "hamper"] for oid in object_ids):
        return EnvironmentType.HOME_LAUNDRY
    if any(keyword in oid for keyword in ["pallet", "racking", "forklift"] for oid in object_ids):
        return EnvironmentType.WAREHOUSE
    if any("bed" in oid or "dresser" in oid for oid in object_ids):
        return EnvironmentType.BEDROOM
    if any("lab" in oid or "bench" in oid or "microscope" in oid for oid in object_ids):
        return EnvironmentType.LAB

    return EnvironmentType.GENERIC


def build_scene_analysis_prompt(
    scene_type: str,
    environment_type: EnvironmentType,
    inventory: dict,
    scene_assets: dict,
    requested_policies: Optional[List[str]] = None,
) -> str:
    objects_summary = []
    for obj in inventory.get("objects", []):
        obj_summary = {
            "id": obj.get("id"),
            "category": obj.get("category"),
            "sim_role": obj.get("sim_role"),
            "description": obj.get("short_description"),
            "articulation": obj.get("articulation_hint"),
            "location": obj.get("approx_location"),
        }
        objects_summary.append(obj_summary)

    available_policies = ENVIRONMENT_DEFAULT_POLICIES.get(
        environment_type, ENVIRONMENT_DEFAULT_POLICIES[EnvironmentType.GENERIC]
    )
    policy_list = "\n".join([f"  - {p.value}" for p in available_policies])

    policy_filter = (
        f"\nUser has specifically requested these policies: {requested_policies}"
        if requested_policies
        else "\nGenerate configurations for ALL applicable policies from the list above."
    )

    prompt = f"""You are an expert in NVIDIA Isaac Sim Replicator for robotics synthetic data generation.

Analyze this scene and generate a comprehensive Replicator configuration for domain randomization.

## Scene Information

Scene Type: {scene_type}
Environment Type: {environment_type.value}

Objects in Scene:
```json
{json.dumps(objects_summary, indent=2)}
```

## Available Policies for {environment_type.value} environments:
{policy_list}
{policy_filter}

## Your Task

Generate a JSON configuration with the schema shown below (placement_regions, variation_assets, policy_configs).
Return ONLY valid JSON in the requested structure.
"""
    return prompt


def analyze_scene_with_gemini(
    client,
    scene_type: str,
    environment_type: EnvironmentType,
    inventory: dict,
    scene_assets: dict,
    requested_policies: Optional[List[str]] = None,
) -> dict:
    prompt = build_scene_analysis_prompt(
        scene_type=scene_type,
        environment_type=environment_type,
        inventory=inventory,
        scene_assets=scene_assets,
        requested_policies=requested_policies,
    )

    print("[REPLICATOR] Calling Gemini for scene analysis...")

    response = client.models.generate_content(
        model="gemini-3.0-pro",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=16000,
            response_mime_type="application/json",
        ),
    )

    response_text = response.text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    try:
        result = json.loads(response_text)
        print("[REPLICATOR] Successfully parsed Gemini response")
        return result
    except json.JSONDecodeError as exc:  # pragma: no cover - fallback path
        print(f"[REPLICATOR] WARNING: Failed to parse Gemini response: {exc}", file=sys.stderr)
        print(f"[REPLICATOR] Response was: {response_text[:500]}...", file=sys.stderr)
        return {
            "analysis": {"scene_summary": "Analysis failed", "recommended_policies": []},
            "placement_regions": [],
            "variation_assets": [],
            "policy_configs": [],
        }


def generate_placement_regions_usda(regions: List[PlacementRegion], scene_id: str) -> str:
    usda_content = f'''#usda 1.0
(
    defaultPrim = "PlacementRegions"
    metersPerUnit = 1.0
    upAxis = "Y"
    doc = "Placement regions for Replicator domain randomization - Scene: {scene_id}"
)

def Xform "PlacementRegions" (
    doc = "Container for all placement regions used by Replicator scripts"
)
{{
'''

    for region in regions:
        pos = region.position or [0, 0, 0]
        size = region.size or [1, 1, 0.01]
        rot = region.rotation or [0, 0, 0]
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", region.name)

        if region.surface_type == "volume":
            geom_type = "Cube"
            extent_attr = f'float3[] extent = [({-size[0]/2}, {-size[1]/2}, {-size[2]/2}), ({size[0]/2}, {size[1]/2}, {size[2]/2})]'
            size_attr = 'double size = 1.0'
            scale_attr = f'double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})'
        else:
            geom_type = "Plane"
            extent_attr = f'float3[] extent = [({-size[0]/2}, 0, {-size[1]/2}), ({size[0]/2}, 0, {size[1]/2})]'
            size_attr = f'double length = {size[0]}\n        double width = {size[1]}'
            scale_attr = 'double3 xformOp:scale = (1, 1, 1)'

        tags_str = ", ".join([f'"{t}"' for t in region.semantic_tags])
        suitable_str = ", ".join([f'"{s}"' for s in region.suitable_for])

        usda_content += f'''
    def Xform "{safe_name}" (
        doc = "{region.description}"
        customData = {{
            string replicator:region_type = "{region.surface_type}"
            string replicator:parent_object = "{region.parent_object_id or ''}"
        }}
    )
    {{
        double3 xformOp:translate = ({pos[0]}, {pos[1]}, {pos[2]})
        double3 xformOp:rotateXYZ = ({rot[0]}, {rot[1]}, {rot[2]})
        {scale_attr}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        custom string[] replicator:semantic_tags = [{tags_str}]
        custom string[] replicator:suitable_for = [{suitable_str}]
        custom string replicator:surface_type = "{region.surface_type}"

        def {geom_type} "Surface" (
            purpose = "guide"
        )
        {{
            {size_attr}
            {extent_attr}
            token visibility = "invisible"
            bool doubleSided = true
            color3f[] primvars:displayColor = [(0.2, 0.8, 0.2)]
            float[] primvars:displayOpacity = [0.3]
        }}
    }}
'''

    usda_content += "}\n"
    return usda_content


def generate_replicator_script(
    policy_config: PolicyConfig,
    all_regions: List[PlacementRegion],
    all_assets: List[VariationAsset],
    scene_id: str,
) -> str:
    region_name_set = set(policy_config.placement_regions)
    regions_used = [r for r in all_regions if not region_name_set or r.name in region_name_set]

    asset_name_set = set(policy_config.variation_assets)
    assets_used = [a for a in all_assets if not asset_name_set or a.name in asset_name_set]

    region_paths = {}
    for region in regions_used:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", region.name)
        region_paths[region.name] = f"/PlacementRegions/{safe_name}/Surface"

    asset_paths = {}
    for asset in assets_used:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", asset.name)
        asset_paths[asset.name] = f"./variation_assets/{safe_name}.usdz"

    randomizers = [asdict(r) for r in policy_config.randomizers]
    capture = policy_config.capture_config or {
        "resolution": [1280, 720],
        "annotations": [
            "rgb",
            "depth",
            "semantic_segmentation",
            "instance_segmentation",
            "bounding_box_2d",
        ],
        "frames_per_episode": 100,
    }

    template = Template(
        """#!/usr/bin/env python3
\"\"\"
Replicator Script: $policy_name
Scene: $scene_id
Policy: $policy_target

$policy_description

This script is auto-generated by the BlueprintRecipe replicator-job.
To use: Open scene.usda in Isaac Sim, then run this script in the Script Editor.
\"\"\"

import omni.replicator.core as rep
from typing import List
import random

SCENE_ROOT = "/World"
PLACEMENT_REGIONS_ROOT = "/PlacementRegions"

PLACEMENT_REGIONS = $placement_regions
VARIATION_ASSETS = $variation_assets
CAPTURE_CONFIG = $capture_config
RANDOMIZER_CONFIGS = $randomizer_configs


def get_all_surfaces():
    surfaces = []
    for name, path in PLACEMENT_REGIONS.items():
        surface = rep.get.prim_at_path(path)
        if surface:
            surfaces.append(surface)
    return surfaces


def load_variation_assets() -> List[str]:
    return list(VARIATION_ASSETS.values())


def create_object_scatter_randomizer(surfaces, asset_paths: List[str], min_objects: int, max_objects: int):
    def randomize():
        num_objects = random.randint(min_objects, max_objects)
        if not asset_paths:
            print("[REPLICATOR] Warning: No assets provided for scatter")
            return None
        objects = rep.create.from_usd(
            rep.distribution.choice(asset_paths, num_objects),
            semantics=[("class", "object")],
            count=num_objects,
        )
        with objects:
            rep.modify.pose(rotation=rep.distribution.uniform((0, -15, 0), (0, 15, 360)))
            if surfaces:
                rep.randomizer.scatter_2d(surface_prims=surfaces, check_for_collisions=True)
        return objects

    return randomize


def create_lighting_randomizer(intensity_range=(0.5, 1.5), color_temp_range=(4000, 6500)):
    def randomize():
        lights = rep.get.prims(path_pattern="/World/.*[Ll]ight.*")
        with lights:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(*intensity_range))
    return randomize


def setup_replicator():
    print("[REPLICATOR] Setting up $policy_name...")
    with rep.new_layer():
        surfaces = get_all_surfaces()
        asset_paths = load_variation_assets()
        resolution = tuple(CAPTURE_CONFIG.get("resolution", [1280, 720]))
        render_product = rep.create.render_product("/World/Camera", resolution)

        registered = []
        for config in RANDOMIZER_CONFIGS:
            if not config.get("enabled", True):
                continue
            name = config.get("name")
            params = config.get("parameters", {})
            if name == "object_scatter":
                randomizer = create_object_scatter_randomizer(
                    surfaces,
                    asset_paths,
                    params.get("min_objects", 5),
                    params.get("max_objects", 15),
                )
                rep.randomizer.register(randomizer)
                registered.append((randomizer, config.get("frequency", "per_frame")))
            elif name == "lighting_variation":
                randomizer = create_lighting_randomizer(
                    tuple(params.get("intensity_range", [0.5, 1.5])),
                    tuple(params.get("color_temperature_range", [4000, 6500])),
                )
                rep.randomizer.register(randomizer)
                registered.append((randomizer, config.get("frequency", "per_episode")))
        return registered, render_product


def run_replicator(num_frames: int | None = None):
    if num_frames is None:
        num_frames = CAPTURE_CONFIG.get("frames_per_episode", 100)
    print(f"[REPLICATOR] Starting data generation: {num_frames} frames")
    randomizers, render_product = setup_replicator()
    with rep.trigger.on_frame(num_frames=num_frames):
        for randomizer, frequency in randomizers:
            if frequency == "per_frame":
                randomizer()
    print("[REPLICATOR] Data generation complete!")


if __name__ == "__main__":
    run_replicator()
else:
    print("[REPLICATOR] Script loaded. Call run_replicator() to start.")
    print("[REPLICATOR] Policy: $policy_name")
    print(f"[REPLICATOR] Regions: {list(PLACEMENT_REGIONS.keys())}")
    print(f"[REPLICATOR] Assets: {len(VARIATION_ASSETS)} variation assets configured")
"""
    )

    return template.substitute(
        policy_name=policy_config.policy_name,
        policy_target=policy_config.policy_target.value,
        policy_description=policy_config.description,
        scene_id=scene_id,
        placement_regions=json.dumps(region_paths, indent=4),
        variation_assets=json.dumps(asset_paths, indent=4),
        capture_config=json.dumps(capture, indent=4),
        randomizer_configs=json.dumps(randomizers, indent=4),
    )


def generate_master_replicator_script(bundle: ReplicatorBundle, policies: List[PolicyConfig]) -> str:
    policies_dict = {policy.policy_id: policy.policy_name for policy in policies}
    policies_json = json.dumps(policies_dict, indent=4)

    template = Template(
        """#!/usr/bin/env python3
\"\"\"
Master Replicator Script for Scene: $scene_id
Environment Type: $environment_type
\"\"\"

import importlib.util
import sys
from pathlib import Path
from typing import Optional, List

AVAILABLE_POLICIES = $available_policies
SCENE_ID = "$scene_id"
ENVIRONMENT_TYPE = "$environment_type"


class ReplicatorManager:
    def __init__(self, scripts_dir: Optional[str] = None):
        self.scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).parent / "policies"

    def list_policies(self) -> List[str]:
        print(f"\nAvailable policies for {SCENE_ID} ({ENVIRONMENT_TYPE}):")
        for policy_id, policy_name in AVAILABLE_POLICIES.items():
            print(f"  {policy_id}: {policy_name}")
        return list(AVAILABLE_POLICIES.keys())

    def _load_module(self, policy_id: str):
        script_path = self.scripts_dir / f"{policy_id}.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Policy script not found: {script_path}")
        spec = importlib.util.spec_from_file_location(policy_id, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[policy_id] = module
        spec.loader.exec_module(module)
        return module

    def run_policy(self, policy_id: str, num_frames: int = 100):
        if policy_id not in AVAILABLE_POLICIES:
            raise ValueError(f"Unknown policy: {policy_id}")
        module = self._load_module(policy_id)
        if hasattr(module, "run_replicator"):
            module.run_replicator(num_frames=num_frames)
        else:
            raise AttributeError("Policy module missing run_replicator")

    def run_all_policies(self, num_frames_each: int = 100):
        for policy_id in AVAILABLE_POLICIES:
            self.run_policy(policy_id, num_frames=num_frames_each)


def list_policies():
    return ReplicatorManager().list_policies()


def run_policy(policy_id: str, num_frames: int = 100):
    ReplicatorManager().run_policy(policy_id, num_frames=num_frames)


def run_all(num_frames_each: int = 100):
    ReplicatorManager().run_all_policies(num_frames_each)


if __name__ == "__main__":
    print("[REPLICATOR] Master script loaded")
    list_policies()
"""
    )

    return template.substitute(
        scene_id=bundle.scene_id,
        environment_type=bundle.environment_type.value,
        available_policies=policies_json,
    )


def generate_asset_manifest(
    assets: List[VariationAsset],
    scene_id: str,
    scene_type: str = "generic",
    environment_type: Optional[EnvironmentType] = None,
    policies: Optional[List[str]] = None,
) -> dict:
    manifest: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "environment_type": environment_type.value if environment_type else "generic",
        "generated_at": None,
        "total_assets": len(assets),
        "policies": policies or [],
        "by_priority": {"required": [], "recommended": [], "optional": []},
        "by_category": {},
        "assets": [],
        "generation_config": {
            "image_model": "gemini-3-pro-image-preview",
            "default_style": "photorealistic product photography",
            "background": "white studio background",
            "lighting": "soft 3-point studio lighting",
        },
    }

    for asset in assets:
        asset_dict = asdict(asset)
        priority = asset_dict.get("priority", "optional")
        if priority in manifest["by_priority"]:
            manifest["by_priority"][priority].append(asset_dict["name"])
        category = asset_dict.get("category", "other")
        manifest["by_category"].setdefault(category, []).append(asset_dict["name"])
        enriched_asset = _enrich_asset_for_generation(asset_dict, scene_type)
        manifest["assets"].append(enriched_asset)

    return manifest


def _enrich_asset_for_generation(asset_dict: dict, scene_type: str) -> dict:
    enriched = asset_dict.copy()
    category = asset_dict.get("category", "").lower()
    semantic_class = asset_dict.get("semantic_class", "").lower()

    material_hints = {
        "dishes": "ceramic, porcelain, or stoneware",
        "utensils": "stainless steel or silver-plated metal",
        "food": "realistic food textures and colors",
        "groceries": "plastic packaging, cardboard boxes, or metal cans",
        "produce": "natural organic textures with realistic imperfections",
        "bottles": "glass or plastic with appropriate transparency",
        "cans": "aluminum or tin with printed labels",
        "boxes": "cardboard with printed packaging graphics",
        "clothing": "cotton, polyester, or mixed fabric textures",
        "towels": "cotton terry cloth or microfiber texture",
        "tools": "metal with rubber or plastic grips",
        "containers": "plastic, glass, or metal storage containers",
        "electronics": "plastic housing with metal accents",
        "office_supplies": "plastic, metal, or paper materials",
        "lab_equipment": "borosilicate glass, stainless steel, or medical-grade plastic",
    }

    style_hints = {
        "dishes": "clean dinnerware with subtle patterns or solid colors",
        "utensils": "polished cutlery, professional quality",
        "groceries": "retail packaging with brand-like graphics",
        "produce": "fresh market quality with natural variations",
        "clothing": "casual household garments with realistic folds",
        "tools": "hand tools with metal bodies and rubber grips",
        "containers": "storage items with functional lids or seals",
        "electronics": "consumer electronics with simple interfaces",
        "office_supplies": "desk accessories suitable for paperwork",
        "lab_equipment": "scientific instruments with clean finishes",
    }

    enriched.setdefault("physics_hints", {})
    if semantic_class:
        enriched["physics_hints"].setdefault("semantic_class", semantic_class)

    if category in material_hints:
        enriched.setdefault("material_hint", material_hints[category])
    if category in style_hints:
        enriched.setdefault("style_hint", style_hints[category])

    return enriched


def _parse_placement_regions(raw_regions: List[dict]) -> List[PlacementRegion]:
    regions: List[PlacementRegion] = []
    for region in raw_regions:
        regions.append(
            PlacementRegion(
                name=region.get("name", "region"),
                description=region.get("description", ""),
                surface_type=region.get("surface_type", "horizontal"),
                parent_object_id=region.get("parent_object_id"),
                position=region.get("position"),
                size=region.get("size"),
                rotation=region.get("rotation"),
                semantic_tags=region.get("semantic_tags", []),
                suitable_for=region.get("suitable_for", []),
            )
        )
    return regions


def _parse_variation_assets(raw_assets: List[dict]) -> List[VariationAsset]:
    assets: List[VariationAsset] = []
    for asset in raw_assets:
        assets.append(
            VariationAsset(
                name=asset.get("name", asset.get("id", "asset")),
                category=asset.get("category", "other"),
                description=asset.get("description", ""),
                semantic_class=asset.get("semantic_class", asset.get("category", "object")),
                priority=asset.get("priority", "optional"),
                source_hint=asset.get("source_hint"),
                example_variants=asset.get("example_variants", []),
                physics_hints=asset.get("physics_hints", {}),
            )
        )
    return assets


def _parse_policies(raw_policies: List[dict]) -> List[PolicyConfig]:
    policies: List[PolicyConfig] = []
    for policy in raw_policies:
        target_value = policy.get("policy_id") or policy.get("policy_target")
        try:
            policy_target = PolicyTarget(target_value)
        except Exception:
            policy_target = PolicyTarget.GENERAL_MANIPULATION

        randomizers = [
            RandomizerConfig(
                name=rand.get("name", ""),
                enabled=rand.get("enabled", True),
                frequency=rand.get("frequency", "per_frame"),
                parameters=rand.get("parameters", {}),
            )
            for rand in policy.get("randomizers", [])
        ]

        policies.append(
            PolicyConfig(
                policy_id=policy.get("policy_id", policy_target.value),
                policy_name=policy.get("policy_name", policy.get("policy_id", "Policy")),
                policy_target=policy_target,
                description=policy.get("description", ""),
                placement_regions=policy.get("placement_regions_used", []),
                variation_assets=policy.get("variation_assets_used", []),
                randomizers=randomizers,
                capture_config=policy.get("capture_config", {}),
                scene_modifications=policy.get("scene_modifications", {}),
            )
        )
    return policies


def _build_bundle_from_analysis(
    scene_id: str,
    scene_type: str,
    environment_type: EnvironmentType,
    analysis_result: dict,
) -> ReplicatorBundle:
    placement_regions = _parse_placement_regions(analysis_result.get("placement_regions", []))
    variation_assets = _parse_variation_assets(analysis_result.get("variation_assets", []))
    policies = _parse_policies(analysis_result.get("policy_configs", []))

    metadata = analysis_result.get("analysis", {})
    if not metadata.get("recommended_policies"):
        metadata["recommended_policies"] = [p.policy_id for p in policies]

    return ReplicatorBundle(
        scene_id=scene_id,
        environment_type=environment_type,
        scene_type=scene_type,
        policies=policies,
        global_placement_regions=placement_regions,
        global_variation_assets=variation_assets,
        metadata=metadata,
    )


def _write_bundle_outputs(
    bundle: ReplicatorBundle,
    output_root: Path,
    scene_id: str,
):
    ensure_dir(output_root)
    ensure_dir(output_root / "policies")
    ensure_dir(output_root / "configs")
    ensure_dir(output_root / "variation_assets")

    # Placement regions layer
    usda_content = generate_placement_regions_usda(bundle.global_placement_regions, scene_id)
    (output_root / "placement_regions.usda").write_text(usda_content, encoding="utf-8")

    # Policy configs and scripts
    for policy in bundle.policies:
        policy_config_path = output_root / "configs" / f"{policy.policy_id}.json"
        save_json(policy_config_path, asdict(policy))
        script_content = generate_replicator_script(policy, bundle.global_placement_regions, bundle.global_variation_assets, scene_id)
        (output_root / "policies" / f"{policy.policy_id}.py").write_text(script_content, encoding="utf-8")

    # Master script
    master_script = generate_master_replicator_script(bundle, bundle.policies)
    (output_root / "replicator_master.py").write_text(master_script, encoding="utf-8")

    # Asset manifest
    manifest = generate_asset_manifest(
        bundle.global_variation_assets,
        scene_id=scene_id,
        scene_type=bundle.scene_type,
        environment_type=bundle.environment_type,
        policies=[p.policy_id for p in bundle.policies],
    )
    save_json(output_root / "variation_assets" / "manifest.json", manifest)

    # Bundle metadata
    bundle_metadata = {
        "scene_id": scene_id,
        "environment_type": bundle.environment_type.value,
        "scene_type": bundle.scene_type,
        "policies": [p.policy_id for p in bundle.policies],
        "placement_regions": [r.name for r in bundle.global_placement_regions],
        "variation_assets": [a.name for a in bundle.global_variation_assets],
        "analysis": bundle.metadata,
    }
    save_json(output_root / "bundle_metadata.json", bundle_metadata)

    # Copy readme for convenience
    repo_readme = Path(__file__).parent / "README.md"
    if repo_readme.exists():
        (output_root / "README.md").write_text(repo_readme.read_text(encoding="utf-8"), encoding="utf-8")



def _load_scene_inputs(scene_root: Path, scene_id: str) -> Tuple[str, dict, dict]:
    inventory_path = scene_root / os.getenv("INVENTORY_PATH", f"scenes/{scene_id}/inventory.json")
    scene_assets_path = scene_root / os.getenv("SCENE_ASSETS_PATH", f"scenes/{scene_id}/assets/manifest.json")
    inventory = load_json(inventory_path) if inventory_path.exists() else {"objects": [], "scene_type": "generic"}
    scene_assets = load_json(scene_assets_path) if scene_assets_path.exists() else {}
    scene_type = inventory.get("scene_type", "generic")
    return scene_type, inventory, scene_assets


def main():
    scene_id = os.getenv("SCENE_ID")
    if not scene_id:
        raise ValueError("SCENE_ID environment variable is required")

    bucket = os.getenv("BUCKET")
    requested_policies = os.getenv("REQUESTED_POLICIES")
    requested = [p.strip() for p in requested_policies.split(",") if p.strip()] if requested_policies else None

    scene_root = GCS_ROOT / bucket if bucket else Path.cwd()
    scene_type, inventory, scene_assets = _load_scene_inputs(scene_root, scene_id)
    environment_type = detect_environment_type(scene_type, inventory)

    analysis_result: Dict[str, Any]
    if os.getenv("GEMINI_API_KEY"):
        try:
            client = create_gemini_client()
            analysis_result = analyze_scene_with_gemini(
                client,
                scene_type=scene_type,
                environment_type=environment_type,
                inventory=inventory,
                scene_assets=scene_assets,
                requested_policies=requested,
            )
        except Exception as exc:  # pragma: no cover - best-effort Gemini path
            print(f"[REPLICATOR] Gemini analysis failed: {exc}")
            analysis_result = {
                "analysis": {"scene_summary": "Gemini unavailable", "recommended_policies": []},
                "placement_regions": [],
                "variation_assets": [],
                "policy_configs": [],
            }
    else:
        print("[REPLICATOR] GEMINI_API_KEY not set; falling back to defaults")
        analysis_result = {
            "analysis": {"scene_summary": "Gemini not configured", "recommended_policies": []},
            "placement_regions": [],
            "variation_assets": [],
            "policy_configs": [],
        }

    # Ensure at least default policies are included
    if not analysis_result.get("policy_configs"):
        default_policies = ENVIRONMENT_DEFAULT_POLICIES.get(environment_type, [])
        analysis_result["policy_configs"] = [
            {
                "policy_id": p.value,
                "policy_name": p.value.replace("_", " ").title(),
                "description": f"Auto-generated {p.value} policy",
                "placement_regions_used": [r.get("name") for r in analysis_result.get("placement_regions", [])],
                "variation_assets_used": [a.get("name") for a in analysis_result.get("variation_assets", [])],
                "randomizers": [],
                "capture_config": {},
            }
            for p in default_policies
        ]

    bundle = _build_bundle_from_analysis(scene_id, scene_type, environment_type, analysis_result)

    output_prefix = os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")
    output_root = scene_root / output_prefix
    _write_bundle_outputs(bundle, output_root, scene_id)

    print(f"[REPLICATOR] Bundle generated at: {output_root}")


if __name__ == "__main__":
    main()
