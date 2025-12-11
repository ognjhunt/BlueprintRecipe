"""Adapter to convert scene plans and matched assets into a canonical scene manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SceneManifestAdapter:
    """Builds the Scene Manifest consumed by downstream USD/Replicator jobs."""

    asset_root: str
    meters_per_unit: float
    coordinate_frame: str = "y_up"
    up_axis: str = "Y"

    def build_manifest(
        self,
        scene_plan: dict[str, Any],
        matched_assets: dict[str, dict[str, Any]],
        recipe: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct a manifest from scene plan, matches, and compiled recipe."""

        inventory = {obj.get("id"): obj for obj in scene_plan.get("object_inventory", [])}
        placements = {p.get("object_id"): p for p in scene_plan.get("spatial_layout", {}).get("placements", [])}
        placement_regions = self._index_regions(scene_plan.get("spatial_layout", {}).get("functional_zones", []))
        relationships = scene_plan.get("spatial_layout", {}).get("relationships", [])

        manifest_objects = []
        for obj in recipe.get("objects", []):
            obj_id = obj.get("id")
            original = inventory.get(obj_id, {})
            match = matched_assets.get(obj_id, {})
            placement = placements.get(obj_id, {})

            manifest_obj = {
                "id": obj_id,
                "name": original.get("label") or original.get("name"),
                "logical_type": obj.get("logical_type") or original.get("category"),
                "category": obj.get("category"),
                "description": original.get("description"),
                "sim_role": self._derive_sim_role(original),
                "placement_region": placement_regions.get(obj_id),
                "transform": self._build_transform(obj.get("transform", {}), placement),
                "asset": self._build_asset_block(match, obj),
                "semantics": obj.get("semantics", {}),
                "physics": obj.get("physics", {}),
                "physics_hints": self._physics_hints(original, match),
                "relationships": self._relationships_for(obj_id, relationships),
                "variation_candidate": bool(original.get("is_variation_candidate")),
            }

            if obj.get("articulation"):
                manifest_obj["articulation"] = obj.get("articulation")

            manifest_objects.append(self._drop_nones(manifest_obj))

        env_analysis = scene_plan.get("environment_analysis", {})
        dims = env_analysis.get("estimated_dimensions", {})

        source_block = {"scene_plan_version": scene_plan.get("version")}
        if metadata:
            if metadata.get("recipe_id") is not None:
                source_block["recipe_id"] = metadata.get("recipe_id")
            if metadata.get("description") is not None:
                source_block["description"] = metadata.get("description")

        manifest = {
            "version": "1.0.0",
            "source": self._drop_nones(source_block),
            "scene": {
                "coordinate_frame": self.coordinate_frame,
                "meters_per_unit": self.meters_per_unit,
                "up_axis": self.up_axis,
                "environment_type": env_analysis.get("detected_type", "unknown"),
                "room_dimensions": {
                    "width": dims.get("width"),
                    "depth": dims.get("depth"),
                    "height": dims.get("height"),
                    "unit": "meters",
                },
                "physics_defaults": {
                    "gravity": {"x": 0.0, "y": -9.81, "z": 0.0},
                    "solver": "physx",
                    "time_steps_per_second": 60,
                },
            },
            "assets": {
                "asset_root": self.asset_root,
                "packs": self._determine_required_packs(matched_assets),
            },
            "objects": manifest_objects,
        }

        return manifest

    def _determine_required_packs(self, matched_assets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        packs = set()
        for asset_info in matched_assets.values():
            if asset_info.get("pack_name"):
                packs.add(asset_info["pack_name"])

        return [
            {"name": pack, "version_hash": "auto", "local_path_template": "${ASSET_ROOT}/" + pack}
            for pack in sorted(packs)
        ]

    def _derive_sim_role(self, obj: dict[str, Any]) -> str:
        if obj.get("is_articulated"):
            return "articulated"
        if obj.get("is_manipulable"):
            return "interactive"
        if obj.get("is_variation_candidate"):
            return "clutter"
        return "static"

    def _build_asset_block(self, matched: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": obj.get("chosen_asset", {}).get("asset_path") or matched.get("chosen_path", ""),
            "pack": matched.get("pack_name"),
            "variants": obj.get("chosen_asset", {}).get("variant_selections") or matched.get("variants", {}),
            "candidates": matched.get("candidates", []),
            "simready_metadata": matched.get("simready_metadata") or matched.get("asset_index_entry") or {},
        }

    def _build_transform(self, transform: dict[str, Any], placement: dict[str, Any]) -> dict[str, Any]:
        # Ensure reference from the original placement is preserved for downstream alignment
        position = dict(transform.get("position", {}))
        reference = placement.get("position", {}).get("reference")
        if reference:
            position["reference"] = reference

        return {
            "position": position,
            "rotation": transform.get("rotation", {}),
            "scale": transform.get("scale", {}),
        }

    def _relationships_for(self, obj_id: str, relationships: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            rel
            for rel in relationships
            if rel.get("subject_id") == obj_id or rel.get("object_id") == obj_id
        ]

    def _index_regions(self, zones: Iterable[dict[str, Any]]) -> dict[str, str]:
        region_map: dict[str, str] = {}
        for zone in zones or []:
            zone_id = zone.get("name") or zone.get("type")
            for obj_id in zone.get("object_ids", []) or []:
                region_map[obj_id] = zone_id
        return region_map

    def _physics_hints(self, obj: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        if obj.get("estimated_dimensions"):
            hints["estimated_dimensions"] = obj["estimated_dimensions"]

        if matched.get("simready_metadata"):
            hints["simready"] = True

        candidates = matched.get("candidates") or []
        if candidates:
            dims = candidates[0].get("dimensions")
            if dims:
                hints.setdefault("estimated_dimensions", dims)

        if obj.get("articulation_type"):
            hints["articulation_type"] = obj.get("articulation_type")

        return hints

    def save_manifest(self, manifest: dict[str, Any], output_path: str | Path) -> Path:
        """Persist the manifest to disk."""
        manifest_path = Path(output_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    def _drop_nones(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove keys with ``None`` values to satisfy schema validation."""

        return {k: v for k, v in data.items() if v is not None}

