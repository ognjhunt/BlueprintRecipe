"""
Scene Planner - Generates scene plans from images using Gemini.

This module uses Gemini to analyze images and generate structured
scene plans that can be compiled into USD recipes.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
        policy_config: Optional[dict[str, Any]] = None
    ):
        self.client = gemini_client or GeminiClient()
        self.policy_config = policy_config or {}

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

            # Normalize object IDs and update references
            self._assign_deterministic_ids(scene_plan)

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

    def _assign_deterministic_ids(self, scene_plan: dict[str, Any]) -> None:
        """Assign deterministic IDs to objects and update spatial references."""

        object_inventory = scene_plan.get("object_inventory", []) or []
        spatial_layout = scene_plan.get("spatial_layout", {}) or {}

        id_map: dict[str, str] = {}
        assigned_ids: set[str] = set()

        for index, obj in enumerate(object_inventory):
            original_id = obj.get("id")
            category = obj.get("category", "object")
            description = obj.get("description", "")

            base_string = f"{category}|{description}|{index}"
            hash_digest = hashlib.sha256(base_string.encode("utf-8")).hexdigest()[:12]
            base_id = f"{category}_{hash_digest}" if category else hash_digest

            new_id = base_id
            suffix = 1
            while new_id in assigned_ids:
                new_id = f"{base_id}_{suffix}"
                suffix += 1

            if original_id and original_id != new_id:
                logging.warning(
                    "Replacing object id '%s' with deterministic id '%s'", original_id, new_id
                )
                id_map[original_id] = new_id
            obj["id"] = new_id
            assigned_ids.add(new_id)

        if not id_map:
            return

        placements = spatial_layout.get("placements", []) or []
        for placement in placements:
            original_obj_id = placement.get("object_id")
            if original_obj_id in id_map:
                placement["object_id"] = id_map[original_obj_id]

        relationships = spatial_layout.get("relationships", []) or []
        for relationship in relationships:
            for key in ("subject_id", "object_id"):
                original_rel_id = relationship.get(key)
                if original_rel_id in id_map:
                    relationship[key] = id_map[original_rel_id]

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
