"""
Scene Planner - Generates scene plans from images using Gemini.

This module uses Gemini to analyze images and generate structured
scene plans that can be compiled into USD recipes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.asset_catalog.catalog_builder import AssetCatalog, AssetCatalogBuilder

from .gemini_client import GeminiClient


@dataclass
class PlanningResult:
    """Result of scene planning."""
    success: bool
    scene_plan: Optional[dict[str, Any]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None


class ScenePlanner:
    """
    Plans scenes from images using Gemini.

    The planner takes an image (photo or sketch) and generates a
    structured scene plan following the ScenePlan schema.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        policy_config: Optional[dict[str, Any]] = None,
        asset_catalog_path: Optional[str] = None,
        asset_catalog: Optional[AssetCatalog] = None
    ):
        self.client = gemini_client or GeminiClient()
        self.policy_config = policy_config or {}
        self.asset_catalog = asset_catalog or self._load_asset_catalog(asset_catalog_path)
        self._articulation_hints = self._build_articulation_hints()

    def plan_from_image(
        self,
        image_path: str,
        task_intent: Optional[str] = None,
        environment_hint: Optional[str] = None,
        target_policies: Optional[list[str]] = None
    ) -> PlanningResult:
        """
        Generate a scene plan from an image.

        Args:
            image_path: Path to the input image
            task_intent: What the scene should enable (e.g., "robot picking mugs")
            environment_hint: Hint about environment type (e.g., "kitchen")
            target_policies: List of target training policies

        Returns:
            PlanningResult with the generated scene plan
        """
        # Build the prompt
        prompt = self._build_planning_prompt(
            task_intent,
            environment_hint,
            target_policies
        )

        # Load and encode image
        try:
            image_data = self._load_image(image_path)
        except Exception as e:
            return PlanningResult(
                success=False,
                error=f"Failed to load image: {str(e)}"
            )

        # Call Gemini
        try:
            response = self.client.generate_with_image(
                prompt=prompt,
                image_data=image_data,
                response_schema=self._get_response_schema()
            )

            # Parse the response
            scene_plan = self._parse_response(response)
            articulation_warnings = self._refine_articulations(scene_plan)
            if articulation_warnings:
                scene_plan.setdefault("warnings", []).extend(articulation_warnings)

            return PlanningResult(
                success=True,
                scene_plan=scene_plan,
                raw_response=response,
                model_used=self.client.model_name
            )

        except Exception as e:
            return PlanningResult(
                success=False,
                error=f"Planning failed: {str(e)}"
            )

    def _build_planning_prompt(
        self,
        task_intent: Optional[str],
        environment_hint: Optional[str],
        target_policies: Optional[list[str]]
    ) -> str:
        """Build the prompt for scene planning."""
        prompt_parts = [
            "Analyze this image and generate a structured scene plan for simulation.",
            "",
            "Your task is to:",
            "1. Identify the environment type (kitchen, warehouse, office, etc.)",
            "2. List all objects visible in the scene with their categories",
            "3. Estimate spatial layout and object relationships",
            "4. Identify articulated objects (doors, drawers, etc.)",
            "5. Suggest training policies appropriate for this scene",
            "",
        ]

        if task_intent:
            prompt_parts.append(f"Task intent: {task_intent}")
            prompt_parts.append("")

        if environment_hint:
            prompt_parts.append(f"Environment hint: {environment_hint}")
            prompt_parts.append("")

        if target_policies:
            prompt_parts.append(f"Target policies: {', '.join(target_policies)}")
            prompt_parts.append("")

        # Add available policies info
        if self.policy_config.get("policies"):
            policy_names = list(self.policy_config["policies"].keys())
            prompt_parts.append(f"Available training policies: {', '.join(policy_names)}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Return a JSON object with the following structure:",
            "- environment_analysis: detected type, confidence, estimated dimensions",
            "- object_inventory: list of all objects with categories and properties",
            "- spatial_layout: placements and relationships between objects",
            "- suggested_policies: recommended training policies",
            "",
            "Be specific about object dimensions and positions where possible.",
            "Mark objects that are articulated (have moving parts).",
            "Identify objects suitable for manipulation training.",
        ])

        return "\n".join(prompt_parts)

    def _load_image(self, image_path: str) -> bytes:
        """Load image data from file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            return f.read()

    def _get_response_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the response."""
        return {
            "type": "object",
            "required": ["environment_analysis", "object_inventory", "spatial_layout"],
            "properties": {
                "environment_analysis": {
                    "type": "object",
                    "properties": {
                        "detected_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "estimated_dimensions": {
                            "type": "object",
                            "properties": {
                                "width": {"type": "number"},
                                "depth": {"type": "number"},
                                "height": {"type": "number"}
                            }
                        }
                    }
                },
                "object_inventory": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "category": {"type": "string"},
                            "description": {"type": "string"},
                            "is_articulated": {"type": "boolean"},
                            "is_manipulable": {"type": "boolean"}
                        }
                    }
                },
                "spatial_layout": {
                    "type": "object",
                    "properties": {
                        "placements": {"type": "array"},
                        "relationships": {"type": "array"}
                    }
                },
                "suggested_policies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "policy_id": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "rationale": {"type": "string"}
                        }
                    }
                }
            }
        }

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the Gemini response into a scene plan."""
        # Try to extract JSON from the response
        try:
            # Handle cases where JSON is wrapped in markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            scene_plan = json.loads(json_str)

            # Add version
            scene_plan["version"] = "1.0.0"

            return scene_plan

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def _load_asset_catalog(self, asset_catalog_path: Optional[str]) -> Optional[AssetCatalog]:
        """Load the asset catalog if available."""
        default_path = Path(__file__).resolve().parents[2] / "asset_index.json"
        catalog_path = Path(asset_catalog_path) if asset_catalog_path else default_path

        if not catalog_path.exists():
            return None

        try:
            return AssetCatalogBuilder.load(str(catalog_path))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: failed to load asset catalog from {catalog_path}: {exc}")
            return None

    def _build_articulation_hints(self) -> dict[str, dict[str, str]]:
        """Build articulation hint map from catalog metadata."""
        if not self.asset_catalog:
            return {}

        hint_map: dict[str, dict[str, str]] = {}
        keyword_defaults = {
            "door": {"articulation_type": "door", "articulation_axis": "y"},
            "drawer": {"articulation_type": "drawer", "articulation_axis": "z"},
            "hinge": {"articulation_type": "door", "articulation_axis": "y"},
            "hinged": {"articulation_type": "door", "articulation_axis": "y"},
            "cabinet": {"articulation_type": "door", "articulation_axis": "y"},
            "cupboard": {"articulation_type": "door", "articulation_axis": "y"},
            "fridge": {"articulation_type": "door", "articulation_axis": "y"},
            "refrigerator": {"articulation_type": "door", "articulation_axis": "y"},
            "oven": {"articulation_type": "door", "articulation_axis": "y"},
            "dishwasher": {"articulation_type": "door", "articulation_axis": "y"},
            "microwave": {"articulation_type": "door", "articulation_axis": "y"},
            "wardrobe": {"articulation_type": "door", "articulation_axis": "y"},
            "closet": {"articulation_type": "door", "articulation_axis": "y"},
            "knob": {"articulation_type": "knob", "articulation_axis": "z"},
            "lever": {"articulation_type": "lever", "articulation_axis": "x"},
            "lid": {"articulation_type": "lid", "articulation_axis": "x"}
        }

        for asset in self.asset_catalog.assets:
            tokens = set(asset.tags)
            tokens.add(asset.category.lower())
            if asset.subcategory:
                tokens.add(asset.subcategory.lower())
            if asset.display_name:
                tokens.update(asset.display_name.lower().replace("_", " ").replace("-", " ").split())

            for keyword, mapping in keyword_defaults.items():
                if any(keyword in token for token in tokens):
                    # Prefer first mapping per keyword
                    hint_map.setdefault(keyword, mapping)

        # Ensure we always have base defaults even if catalog is sparse
        for keyword, mapping in keyword_defaults.items():
            hint_map.setdefault(keyword, mapping)

        return hint_map

    def _infer_articulation_from_catalog(self, obj: dict[str, Any]) -> Optional[dict[str, str]]:
        """Infer articulation hints from the catalog using object metadata."""
        if not self._articulation_hints:
            return None

        text = " ".join([
            str(obj.get("category", "")),
            str(obj.get("subcategory", "")),
            str(obj.get("description", "")),
        ]).lower()

        for keyword, mapping in self._articulation_hints.items():
            if keyword in text:
                return mapping

        return None

    def _refine_articulations(self, scene_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Adjust articulation flags/types using catalog hints."""
        warnings = []

        objects = scene_plan.get("object_inventory", [])
        for obj in objects:
            hint = self._infer_articulation_from_catalog(obj)
            flagged = obj.get("is_articulated")

            if hint:
                obj["is_articulated"] = True
                obj.setdefault("articulation_type", hint["articulation_type"])
                obj.setdefault("articulation_axis", hint["articulation_axis"])
            elif flagged:
                warnings.append({
                    "type": "articulation_missing_in_catalog",
                    "message": (
                        f"Articulation flagged for {obj.get('id', 'unknown')} but no matching"
                        " articulation keywords were found in the asset catalog."
                    ),
                    "affected_objects": [obj.get("id", "unknown")],
                })

        return warnings

    def validate_plan(self, scene_plan: dict[str, Any]) -> list[str]:
        """Validate a scene plan and return any warnings."""
        warnings = []

        # Check required fields
        if "environment_analysis" not in scene_plan:
            warnings.append("Missing environment_analysis")

        if "object_inventory" not in scene_plan:
            warnings.append("Missing object_inventory")
        elif not scene_plan["object_inventory"]:
            warnings.append("Empty object_inventory")

        if "spatial_layout" not in scene_plan:
            warnings.append("Missing spatial_layout")

        # Check for objects without categories
        for obj in scene_plan.get("object_inventory", []):
            if not obj.get("category"):
                warnings.append(f"Object {obj.get('id', 'unknown')} has no category")

        return warnings
