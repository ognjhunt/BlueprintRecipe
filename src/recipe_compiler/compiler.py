"""
Recipe Compiler - Main orchestrator for converting scene plans to USD recipes.

This module takes a ScenePlan (from Gemini) and matched assets, then compiles
them into a complete recipe package including:
- scene.usda (top-level stage with references)
- recipe.json (the machine-readable recipe)
- Layered USD files for composition
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .layer_manager import LayerManager
from .physics_estimator import PhysicsEstimator
from .usd_builder import USDSceneBuilder


@dataclass
class CompilerConfig:
    """Configuration for the recipe compiler."""
    output_dir: str
    asset_root: str
    coordinate_frame: str = "y_up"  # or "z_up"
    meters_per_unit: float = 1.0
    up_axis: str = "Y"
    include_physics: bool = True
    include_semantics: bool = True
    generate_thumbnails: bool = False
    deterministic_seed: Optional[int] = None


@dataclass
class CompilationResult:
    """Result of recipe compilation."""
    success: bool
    recipe_path: str
    scene_path: str
    layer_paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    qa_report: dict[str, Any] = field(default_factory=dict)


class RecipeCompiler:
    """
    Compiles scene plans and matched assets into USD scene recipes.

    The compiler produces a layered USD structure:
    - scene.usda: Top-level stage (references all layers)
    - layers/room_shell.usda: Room geometry and walls
    - layers/layout.usda: Object placements and transforms
    - layers/semantics.usda: Semantic labels and annotations
    - layers/physics_overrides.usda: Physics properties
    """

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.layer_manager = LayerManager(config.output_dir)
        self.usd_builder = USDSceneBuilder(
            meters_per_unit=config.meters_per_unit,
            up_axis=config.up_axis
        )
        self.physics_estimator = PhysicsEstimator()

    def compile(
        self,
        scene_plan: dict[str, Any],
        matched_assets: dict[str, dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None
    ) -> CompilationResult:
        """
        Compile a scene plan into a complete recipe package.

        Args:
            scene_plan: The AI-generated scene plan (ScenePlan schema)
            matched_assets: Mapping of object_id -> matched asset info
            metadata: Additional metadata for the recipe

        Returns:
            CompilationResult with paths to generated files
        """
        warnings = []
        errors = []

        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        layers_path = output_path / "layers"
        layers_path.mkdir(exist_ok=True)

        try:
            # Step 1: Build the recipe JSON
            recipe = self._build_recipe(scene_plan, matched_assets, metadata)
            recipe_path = output_path / "recipe.json"
            with open(recipe_path, "w") as f:
                json.dump(recipe, f, indent=2)

            # Step 2: Build USD layers
            layer_paths = self._build_usd_layers(
                recipe, scene_plan, matched_assets, layers_path
            )

            # Step 3: Build top-level scene.usda
            scene_path = self._build_scene_stage(
                output_path, layer_paths, recipe
            )

            # Step 4: Validate and generate QA report
            qa_report = self._validate_compilation(
                scene_path, layer_paths, recipe
            )

            qa_report_path = output_path / "qa" / "compilation_report.json"
            qa_report_path.parent.mkdir(exist_ok=True)
            with open(qa_report_path, "w") as f:
                json.dump(qa_report, f, indent=2)

            return CompilationResult(
                success=len(errors) == 0,
                recipe_path=str(recipe_path),
                scene_path=str(scene_path),
                layer_paths={k: str(v) for k, v in layer_paths.items()},
                warnings=warnings,
                errors=errors,
                qa_report=qa_report
            )

        except Exception as e:
            errors.append(f"Compilation failed: {str(e)}")
            return CompilationResult(
                success=False,
                recipe_path="",
                scene_path="",
                errors=errors
            )

    def _build_recipe(
        self,
        scene_plan: dict[str, Any],
        matched_assets: dict[str, dict[str, Any]],
        metadata: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the recipe.json structure."""
        env_type = scene_plan.get("environment_analysis", {}).get("detected_type", "unknown")

        recipe = {
            "version": "1.0.0",
            "metadata": {
                "recipe_id": metadata.get("recipe_id", self._generate_id()) if metadata else self._generate_id(),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "environment_type": env_type,
                "source_image_uri": metadata.get("source_image_uri") if metadata else None,
                "description": metadata.get("description") if metadata else None,
                "deterministic_seed": self.config.deterministic_seed,
                "planning_model": metadata.get("planning_model") if metadata else None,
                "target_policies": scene_plan.get("suggested_policies", [])
            },
            "asset_packs": self._determine_required_packs(matched_assets),
            "asset_resolver": {
                "primary_source": "local",
                "local_root": self.config.asset_root,
                "fallback_sources": ["nucleus"]
            },
            "room": self._extract_room_config(scene_plan),
            "objects": self._build_objects_list(scene_plan, matched_assets),
            "relationships": self._extract_relationships(scene_plan),
            "placement_regions": self._extract_placement_regions(scene_plan)
        }

        return recipe

    def _build_objects_list(
        self,
        scene_plan: dict[str, Any],
        matched_assets: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build the objects list for the recipe."""
        objects = []

        for obj in scene_plan.get("object_inventory", []):
            obj_id = obj.get("id")
            matched = matched_assets.get(obj_id, {})

            # Get placement info
            placement = self._find_placement(obj_id, scene_plan)

            scene_obj = {
                "id": obj_id,
                "logical_type": obj.get("category"),
                "category": obj.get("subcategory", obj.get("category")),
                "candidate_assets": matched.get("candidates", []),
                "chosen_asset": {
                    "asset_path": matched.get("chosen_path", ""),
                    "variant_selections": matched.get("variants", {})
                },
                "transform": {
                    "position": placement.get("position", {"x": 0, "y": 0, "z": 0}),
                    "rotation": placement.get("rotation", {"w": 1, "x": 0, "y": 0, "z": 0}),
                    "scale": placement.get("scale", {"x": 1, "y": 1, "z": 1})
                },
                "semantics": {
                    "class": obj.get("category"),
                    "instance_id": obj_id,
                    "affordances": self._infer_affordances(obj)
                },
                "physics": self._get_physics_config(obj, matched),
                "manipulable": obj.get("is_manipulable", False),
                "variation_asset": obj.get("is_variation_candidate", False)
            }

            # Add articulation if present
            if obj.get("is_articulated"):
                scene_obj["articulation"] = self._get_articulation_config(obj, matched)

            objects.append(scene_obj)

        return objects

    def _build_usd_layers(
        self,
        recipe: dict[str, Any],
        scene_plan: dict[str, Any],
        matched_assets: dict[str, dict[str, Any]],
        layers_path: Path
    ) -> dict[str, Path]:
        """Build the individual USD layers."""
        layer_paths = {}

        # Room shell layer
        room_layer = self.layer_manager.create_layer(
            "room_shell",
            layers_path / "room_shell.usda"
        )
        self.usd_builder.build_room_shell(room_layer, recipe.get("room", {}))
        layer_paths["room_shell"] = layers_path / "room_shell.usda"

        # Layout layer (object placements)
        layout_layer = self.layer_manager.create_layer(
            "layout",
            layers_path / "layout.usda"
        )
        self.usd_builder.build_layout(
            layout_layer,
            recipe.get("objects", []),
            self.config.asset_root
        )
        layer_paths["layout"] = layers_path / "layout.usda"

        # Semantics layer
        if self.config.include_semantics:
            semantics_layer = self.layer_manager.create_layer(
                "semantics",
                layers_path / "semantics.usda"
            )
            self.usd_builder.build_semantics(
                semantics_layer,
                recipe.get("objects", [])
            )
            layer_paths["semantics"] = layers_path / "semantics.usda"

        # Physics overrides layer
        if self.config.include_physics:
            physics_layer = self.layer_manager.create_layer(
                "physics_overrides",
                layers_path / "physics_overrides.usda"
            )
            self.usd_builder.build_physics_overrides(
                physics_layer,
                recipe.get("objects", [])
            )
            layer_paths["physics_overrides"] = layers_path / "physics_overrides.usda"

        # Save all layers
        self.layer_manager.save_all()

        return layer_paths

    def _build_scene_stage(
        self,
        output_path: Path,
        layer_paths: dict[str, Path],
        recipe: dict[str, Any]
    ) -> Path:
        """Build the top-level scene.usda that references all layers."""
        scene_path = output_path / "scene.usda"

        # Create the top-level stage
        stage = self.usd_builder.create_stage(str(scene_path))

        # Add sublayer references
        for layer_name, layer_path in layer_paths.items():
            relative_path = f"./layers/{layer_path.name}"
            self.usd_builder.add_sublayer(stage, relative_path)

        # Set stage metadata
        self.usd_builder.set_stage_metadata(
            stage,
            up_axis=self.config.up_axis,
            meters_per_unit=self.config.meters_per_unit,
            doc=f"BlueprintRecipe scene: {recipe['metadata'].get('recipe_id', 'unknown')}"
        )

        # Save the stage
        stage.Save()

        return scene_path

    def _validate_compilation(
        self,
        scene_path: Path,
        layer_paths: dict[str, Path],
        recipe: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate the compiled scene and generate QA report."""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scene_path": str(scene_path),
            "checks": {},
            "warnings": [],
            "errors": []
        }

        # Check file existence
        report["checks"]["scene_exists"] = scene_path.exists()
        report["checks"]["layers_exist"] = all(p.exists() for p in layer_paths.values())

        # Check asset references (would need USD runtime for full validation)
        report["checks"]["asset_references"] = "pending_runtime_validation"

        # Check physics configuration
        physics_objects = [
            obj for obj in recipe.get("objects", [])
            if obj.get("physics", {}).get("enabled", False)
        ]
        report["checks"]["physics_configured"] = len(physics_objects)

        # Check semantics
        semantic_objects = [
            obj for obj in recipe.get("objects", [])
            if obj.get("semantics", {}).get("class")
        ]
        report["checks"]["semantics_configured"] = len(semantic_objects)

        # Check articulations
        articulated_objects = [
            obj for obj in recipe.get("objects", [])
            if obj.get("articulation")
        ]
        report["checks"]["articulations_configured"] = len(articulated_objects)

        return report

    def _generate_id(self) -> str:
        """Generate a unique recipe ID."""
        import uuid
        return f"recipe_{uuid.uuid4().hex[:12]}"

    def _determine_required_packs(
        self,
        matched_assets: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Determine which asset packs are required."""
        packs = set()
        for asset_info in matched_assets.values():
            if "pack_name" in asset_info:
                packs.add(asset_info["pack_name"])

        return [
            {"name": pack, "version_hash": "auto", "local_path_template": "${ASSET_ROOT}/" + pack}
            for pack in packs
        ]

    def _extract_room_config(self, scene_plan: dict[str, Any]) -> dict[str, Any]:
        """Extract room configuration from scene plan."""
        env_analysis = scene_plan.get("environment_analysis", {})
        dims = env_analysis.get("estimated_dimensions", {})

        return {
            "dimensions": {
                "width": dims.get("width", 5.0),
                "depth": dims.get("depth", 5.0),
                "height": dims.get("height", 3.0),
                "unit": "meters"
            },
            "coordinate_frame": self.config.coordinate_frame
        }

    def _find_placement(
        self,
        obj_id: str,
        scene_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Find placement info for an object."""
        spatial = scene_plan.get("spatial_layout", {})
        for placement in spatial.get("placements", []):
            if placement.get("object_id") == obj_id:
                pos = placement.get("position", {})
                return {
                    "position": {
                        "x": pos.get("x", 0),
                        "y": pos.get("y", 0),
                        "z": pos.get("z", 0)
                    },
                    "rotation": self._rotation_from_degrees(
                        placement.get("rotation_degrees", 0)
                    ),
                    "scale": {"x": 1, "y": 1, "z": 1}
                }
        return {}

    def _rotation_from_degrees(self, degrees: float) -> dict[str, float]:
        """Convert rotation in degrees to quaternion (Y-up rotation)."""
        import math
        rad = math.radians(degrees)
        return {
            "w": math.cos(rad / 2),
            "x": 0,
            "y": math.sin(rad / 2),
            "z": 0
        }

    def _infer_affordances(self, obj: dict[str, Any]) -> list[str]:
        """Infer affordances from object properties."""
        affordances = []
        if obj.get("is_manipulable"):
            affordances.append("graspable")
        if obj.get("is_articulated"):
            art_type = obj.get("articulation_type", "")
            if art_type == "door":
                affordances.extend(["openable", "closable"])
            elif art_type == "drawer":
                affordances.extend(["pullable", "pushable"])
            elif art_type == "knob":
                affordances.append("rotatable")
        return affordances

    def _get_physics_config(
        self,
        obj: dict[str, Any],
        matched: dict[str, Any]
    ) -> dict[str, Any]:
        """Get physics configuration for an object."""
        dimensions = self._extract_dimensions(obj, matched)

        config = {
            "enabled": True,
            "collision_enabled": True,
            "rigid_body": obj.get("is_manipulable", False),
        }

        simready = matched.get("simready_metadata", {})
        # Prefer Gemini-sourced physics augmented with priors to avoid hardcoded defaults
        ai_physics = self.physics_estimator.estimate(obj, matched, dimensions)
        if ai_physics:
            config.update(ai_physics)

        # Merge any simready metadata provided by matched assets
        for key in (
            "mass_kg",
            "collision_approximation",
            "center_of_mass_offset",
            "friction_static",
            "friction_dynamic",
            "restitution",
        ):
            if key in simready and simready.get(key) is not None:
                if key == "mass_kg":
                    config["mass_override"] = float(simready[key])
                else:
                    config[key] = simready[key]

        # Preserve explicit user-provided overrides
        if obj.get("mass_override") is not None:
            config["mass_override"] = float(obj.get("mass_override"))

        # Compute inertia if we have a mass value
        inertia = self._compute_inertia_diagonal(dimensions, config.get("mass_override"))
        if inertia:
            config["inertia_diagonal"] = inertia

        return config

    def _extract_dimensions(self, obj: dict[str, Any], matched: dict[str, Any]) -> dict[str, float]:
        """Best-effort extraction of object dimensions in meters."""
        dims = obj.get("estimated_dimensions") or {}
        if dims and all(k in dims for k in ("width", "depth", "height")):
            return {k: float(dims.get(k, 0)) for k in ("width", "depth", "height")}

        # Fall back to matched candidate dimensions
        candidates = matched.get("candidates") or []
        if candidates:
            candidate_dims = candidates[0].get("dimensions") or {}
            if candidate_dims:
                return {
                    "width": float(candidate_dims.get("width", 0)),
                    "depth": float(candidate_dims.get("depth", 0)),
                    "height": float(candidate_dims.get("height", 0)),
                }

        return {}

    def _compute_inertia_diagonal(
        self, dimensions: dict[str, float], mass: Optional[float]
    ) -> Optional[list[float]]:
        """Compute a diagonal inertia approximation for a box."""
        if not dimensions or not mass:
            return None

        width = float(dimensions.get("width", 0))
        depth = float(dimensions.get("depth", 0))
        height = float(dimensions.get("height", 0))

        if min(width, depth, height) <= 0:
            return None

        factor = mass / 12.0
        ix = factor * (depth ** 2 + height ** 2)
        iy = factor * (width ** 2 + height ** 2)
        iz = factor * (width ** 2 + depth ** 2)
        return [ix, iy, iz]

    def _get_articulation_config(
        self,
        obj: dict[str, Any],
        matched: dict[str, Any]
    ) -> dict[str, Any]:
        """Get articulation configuration for an object."""
        art_type = obj.get("articulation_type", "door")

        type_mapping = {
            "door": {"type": "revolute", "axis": "y", "limits": {"lower": 0, "upper": 1.57}},
            "drawer": {"type": "prismatic", "axis": "z", "limits": {"lower": 0, "upper": 0.5}},
            "lid": {"type": "revolute", "axis": "x", "limits": {"lower": 0, "upper": 1.57}},
            "knob": {"type": "revolute", "axis": "z", "limits": {"lower": 0, "upper": 6.28}},
            "lever": {"type": "revolute", "axis": "x", "limits": {"lower": -0.5, "upper": 0.5}}
        }

        return type_mapping.get(art_type, type_mapping["door"])

    def _extract_relationships(self, scene_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract relationships from scene plan."""
        spatial = scene_plan.get("spatial_layout", {})
        return spatial.get("relationships", [])

    def _extract_placement_regions(self, scene_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract placement regions from scene plan."""
        spatial = scene_plan.get("spatial_layout", {})
        zones = spatial.get("functional_zones", [])

        regions = []
        for zone in zones:
            bounds = zone.get("bounds_estimate", {})
            regions.append({
                "id": zone.get("name", "").lower().replace(" ", "_"),
                "type": zone.get("type", "general"),
                "surface_type": "horizontal",
                "bounds": {
                    "min": {"x": bounds.get("min_x", 0), "y": 0, "z": bounds.get("min_y", 0)},
                    "max": {"x": bounds.get("max_x", 1), "y": 0.1, "z": bounds.get("max_y", 1)}
                }
            })

        return regions
