"""
Scene Planner - Generates scene plans from images using Gemini.

This module uses Gemini to analyze images and generate structured
scene plans that can be compiled into USD recipes.
"""

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Pillow is required for image preprocessing. Install with `pip install pillow`."
    ) from exc

from .gemini_client import GeminiClient


@dataclass
class PlanningResult:
    """Result of scene planning."""
    success: bool
    scene_plan: Optional[dict[str, Any]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    warnings: Optional[list[str]] = None


class ScenePlanner:
    """
    Plans scenes from images using Gemini.

    The planner takes an image (photo or sketch) and generates a
    structured scene plan following the ScenePlan schema.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        policy_config: Optional[dict[str, Any]] = None
    ):
        self.client = gemini_client or GeminiClient()
        self.policy_config = policy_config or {}
        self.logger = logging.getLogger(__name__)
        self.asset_catalog = self._load_asset_catalog()
        self.articulation_index = self._build_articulation_index()

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

        # Load and preprocess image
        try:
            image_data = self._preprocess_image(image_path)
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

            warnings: list[str] = []
            warnings.extend(self._assign_deterministic_ids(scene_plan))
            warnings.extend(self._refine_articulations(scene_plan))

            # Validate and clean dimensions
            validation_warnings = self.validate_plan(scene_plan)
            warnings.extend(validation_warnings)

            if warnings:
                scene_plan["validation_warnings"] = warnings

            return PlanningResult(
                success=True,
                scene_plan=scene_plan,
                raw_response=response,
                model_used=self.client.model_name,
                warnings=warnings or None
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

    def _load_asset_catalog(self) -> dict[str, Any]:
        """Load asset catalog metadata for articulation cross-references."""
        catalog_path = Path(__file__).resolve().parents[2] / "asset_index.json"
        if not catalog_path.exists():
            self.logger.debug("Asset catalog not found at %s", catalog_path)
            return {}

        try:
            with catalog_path.open("r", encoding="utf-8") as catalog_file:
                return json.load(catalog_file)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - runtime guard
            self.logger.warning("Failed to load asset catalog: %s", exc)
            return {}

    def _build_articulation_index(self) -> dict[str, Any]:
        """Extract articulation hints from the asset catalog."""
        articulation_keywords = {
            "door": {"type": "hinge", "axis": "y"},
            "doors": {"type": "hinge", "axis": "y"},
            "drawer": {"type": "slide", "axis": "z"},
            "drawers": {"type": "slide", "axis": "z"},
            "hinge": {"type": "hinge", "axis": "y"},
            "hinges": {"type": "hinge", "axis": "y"},
        }

        category_map: dict[str, set[str]] = {}
        if not self.asset_catalog:
            return {"categories": category_map, "keywords": articulation_keywords}

        for asset in self.asset_catalog.get("assets", []):
            searchable_text = " ".join(
                filter(
                    None,
                    [
                        asset.get("category", ""),
                        asset.get("subcategory", ""),
                        asset.get("display_name", ""),
                        " ".join(asset.get("tags", [])),
                    ],
                )
            ).lower()

            for keyword in articulation_keywords:
                if keyword in searchable_text:
                    category = (asset.get("category") or "").lower()
                    if category:
                        category_map.setdefault(category, set()).add(keyword)

        return {"categories": category_map, "keywords": articulation_keywords}

    def _preprocess_image(self, image_path: str) -> bytes:
        """Load and preprocess image data for Gemini."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        max_file_size_bytes = 4 * 1024 * 1024  # ~4MB
        max_dimension = 4096

        try:
            with Image.open(path) as img:
                # Respect EXIF orientation
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                width, height = img.size
                if max(width, height) > max_dimension:
                    scale = max_dimension / float(max(width, height))
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                output = io.BytesIO()
                img.save(output, format="JPEG", quality=85, optimize=True)
                data = output.getvalue()

                if len(data) > max_file_size_bytes:
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=75, optimize=True)
                    data = output.getvalue()

                return data
        except Exception as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(f"Failed to preprocess image: {exc}") from exc

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

    def _assign_deterministic_ids(self, scene_plan: dict[str, Any]) -> list[str]:
        """Assign deterministic IDs and update references in spatial layout."""
        warnings: list[str] = []
        id_map: dict[str, str] = {}
        used_ids: set[str] = set()
        inventory = scene_plan.get("object_inventory") or []

        for index, obj in enumerate(inventory):
            category = obj.get("category") or "object"
            description = obj.get("description") or obj.get("display_name") or ""
            seed = f"{category}:{description}:{index}".lower()
            deterministic_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]

            if deterministic_id in used_ids:
                suffix = 1
                while f"{deterministic_id}-{suffix}" in used_ids:
                    suffix += 1
                deterministic_id = f"{deterministic_id}-{suffix}"

            original_id = obj.get("id")
            if original_id != deterministic_id:
                if original_id:
                    id_map[original_id] = deterministic_id
                    warnings.append(
                        f"Replaced object id {original_id} with deterministic id {deterministic_id}"
                    )
                else:
                    warnings.append(
                        f"Assigned deterministic id {deterministic_id} to object without id"
                    )

            obj["id"] = deterministic_id
            used_ids.add(deterministic_id)

        if id_map and scene_plan.get("spatial_layout"):
            self._update_spatial_layout_references(scene_plan["spatial_layout"], id_map)

        for warning in warnings:
            self.logger.warning(warning)

        return warnings

    def _update_spatial_layout_references(self, spatial_layout: Any, id_map: dict[str, str]) -> None:
        """Recursively update spatial layout references to match new IDs."""
        if isinstance(spatial_layout, dict):
            for key, value in spatial_layout.items():
                if isinstance(value, (dict, list)):
                    self._update_spatial_layout_references(value, id_map)
                elif isinstance(value, str) and value in id_map:
                    spatial_layout[key] = id_map[value]
        elif isinstance(spatial_layout, list):
            for index, item in enumerate(spatial_layout):
                if isinstance(item, (dict, list)):
                    self._update_spatial_layout_references(item, id_map)
                elif isinstance(item, str) and item in id_map:
                    spatial_layout[index] = id_map[item]

    def _refine_articulations(self, scene_plan: dict[str, Any]) -> list[str]:
        """Cross-reference asset metadata to refine articulation flags."""
        warnings: list[str] = []
        inventory = scene_plan.get("object_inventory") or []
        category_map: dict[str, set[str]] = self.articulation_index.get("categories", {})
        keyword_details: dict[str, dict[str, str]] = self.articulation_index.get("keywords", {})

        for obj in inventory:
            matched_keywords: set[str] = set()
            category = (obj.get("category") or "").lower()
            description = (obj.get("description") or "").lower()

            if category and category in category_map:
                matched_keywords.update(category_map[category])

            for keyword in keyword_details:
                if keyword in category or keyword in description:
                    matched_keywords.add(keyword)

            if matched_keywords:
                obj["is_articulated"] = True
                articulation_types = {keyword_details[keyword]["type"] for keyword in matched_keywords if keyword in keyword_details}
                articulation_axes = {
                    keyword_details[keyword]["axis"]
                    for keyword in matched_keywords
                    if keyword in keyword_details and keyword_details[keyword].get("axis")
                }

                if articulation_types:
                    obj["articulation_types"] = sorted(articulation_types)
                if articulation_axes:
                    obj["articulation_axes"] = sorted(articulation_axes)
            elif obj.get("is_articulated"):
                warning = (
                    f"Object {obj.get('id', 'unknown')} marked articulated without catalog support"
                )
                warnings.append(warning)
                self.logger.warning(warning)

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

        warnings.extend(self._validate_dimensions(scene_plan))

        if warnings:
            scene_plan.setdefault("validation_warnings", []).extend(warnings)

        return warnings

    def _validate_dimensions(self, scene_plan: dict[str, Any]) -> list[str]:
        """Validate object dimensions against typical ranges and clamp outliers."""
        warnings = []

        typical_dimensions: dict[str, dict[str, tuple[float, float]]] = {
            "refrigerator": {
                "height": (1.5, 2.2),
                "width": (0.5, 1.0),
                "depth": (0.5, 1.0)
            },
            "fridge": {
                "height": (1.5, 2.2),
                "width": (0.5, 1.0),
                "depth": (0.5, 1.0)
            },
            "mug": {
                "height": (0.07, 0.15),
                "width": (0.07, 0.1),
                "depth": (0.07, 0.1)
            },
            "table": {
                "height": (0.45, 0.9),
                "width": (0.6, 2.0),
                "depth": (0.6, 1.2)
            },
            "chair": {
                "height": (0.75, 1.3),
                "width": (0.35, 0.7),
                "depth": (0.35, 0.7)
            }
        }

        for obj in scene_plan.get("object_inventory", []):
            category = (obj.get("category") or "").lower()
            ranges = typical_dimensions.get(category)
            dimensions = obj.get("dimensions") or obj.get("size")

            if not ranges or not isinstance(dimensions, dict):
                continue

            for dim_key, (min_val, max_val) in ranges.items():
                if dim_key not in dimensions or not isinstance(dimensions[dim_key], (int, float)):
                    continue

                value = dimensions[dim_key]
                if value < min_val:
                    warnings.append(
                        f"{obj.get('id', 'unknown')} {category} {dim_key} {value}m below typical; clamped to {min_val}m"
                    )
                    dimensions[dim_key] = min_val
                elif value > max_val:
                    warnings.append(
                        f"{obj.get('id', 'unknown')} {category} {dim_key} {value}m above typical; clamped to {max_val}m"
                    )
                    dimensions[dim_key] = max_val

        return warnings
