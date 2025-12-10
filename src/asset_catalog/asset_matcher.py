"""
Asset Matcher - Matches scene plan objects to catalog assets.

This module takes object descriptions from the scene plan and finds
the best matching assets from the indexed catalog using:
- Tag-based search
- Dimension matching
- Embedding similarity (when available)
- Style/attribute matching
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .catalog_builder import AssetCatalog, AssetEntry


@dataclass
class AssetMatch:
    """Represents a matched asset with confidence score."""
    asset_id: str
    asset_path: str
    score: float
    match_reasons: list[str]
    dimensions: Optional[dict[str, float]] = None
    variants: Optional[dict[str, str]] = None


@dataclass
class MatchResult:
    """Result of matching an object to assets."""
    object_id: str
    candidates: list[AssetMatch]
    chosen: Optional[AssetMatch] = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AssetMatcher:
    """
    Matches scene plan objects to catalog assets.

    Scoring factors:
    - Category match: +0.3
    - Tag overlap: +0.1 per matching tag (max 0.3)
    - Dimension compatibility: +0.2
    - Style attributes: +0.1 per match
    - Embedding similarity: +0.1 (when available)
    """

    def __init__(
        self,
        catalog: AssetCatalog,
        embeddings_db: Optional[Any] = None
    ):
        self.catalog = catalog
        self.embeddings_db = embeddings_db

        # Build search indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build search indices for fast lookup."""
        self.category_index: dict[str, list[AssetEntry]] = {}
        self.tag_index: dict[str, list[AssetEntry]] = {}

        for asset in self.catalog.assets:
            # Category index
            cat = asset.category.lower()
            if cat not in self.category_index:
                self.category_index[cat] = []
            self.category_index[cat].append(asset)

            # Tag index
            for tag in asset.tags:
                tag_lower = tag.lower()
                if tag_lower not in self.tag_index:
                    self.tag_index[tag_lower] = []
                self.tag_index[tag_lower].append(asset)

    def match(
        self,
        object_spec: dict[str, Any],
        top_k: int = 5,
        dimension_tolerance: float = 0.3
    ) -> MatchResult:
        """
        Match an object specification to catalog assets.

        Args:
            object_spec: Object from scene plan with category, description, etc.
            top_k: Number of top candidates to return
            dimension_tolerance: Fraction tolerance for dimension matching

        Returns:
            MatchResult with ranked candidates
        """
        object_id = object_spec.get("id", "unknown")
        category = object_spec.get("category", "").lower()
        description = object_spec.get("description", "")
        attributes = object_spec.get("attributes", {})
        est_dims = object_spec.get("estimated_dimensions", {})

        candidates = []
        warnings = []

        # Get candidate assets
        candidate_assets = self._get_candidates(category, description)

        if not candidate_assets:
            warnings.append(f"No assets found for category '{category}'")
            return MatchResult(
                object_id=object_id,
                candidates=[],
                warnings=warnings
            )

        # Score each candidate
        for asset in candidate_assets:
            score, reasons = self._score_asset(
                asset, category, description, attributes, est_dims, dimension_tolerance
            )

            candidates.append(AssetMatch(
                asset_id=asset.asset_id,
                asset_path=asset.relative_path,
                score=score,
                match_reasons=reasons,
                dimensions=asset.dimensions,
                variants=self._get_default_variants(asset)
            ))

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Take top k
        candidates = candidates[:top_k]

        # Auto-select best if above threshold
        chosen = None
        if candidates and candidates[0].score >= 0.5:
            chosen = candidates[0]

        return MatchResult(
            object_id=object_id,
            candidates=candidates,
            chosen=chosen,
            warnings=warnings
        )

    def match_batch(
        self,
        objects: list[dict[str, Any]],
        top_k: int = 5
    ) -> dict[str, MatchResult]:
        """Match multiple objects to assets."""
        results = {}
        for obj in objects:
            result = self.match(obj, top_k=top_k)
            results[obj.get("id", "unknown")] = result
        return results

    def _get_candidates(
        self,
        category: str,
        description: str
    ) -> list[AssetEntry]:
        """Get candidate assets based on category and description."""
        candidates = set()

        # First, try exact category match
        if category in self.category_index:
            for asset in self.category_index[category]:
                candidates.add(asset.asset_id)

        # Also search by tags from description
        desc_words = self._tokenize(description)
        for word in desc_words:
            if word in self.tag_index:
                for asset in self.tag_index[word]:
                    candidates.add(asset.asset_id)

        # If still empty, try broader category matching
        if not candidates:
            for cat, assets in self.category_index.items():
                if category in cat or cat in category:
                    for asset in assets:
                        candidates.add(asset.asset_id)

        # Convert to asset entries
        return [
            asset for asset in self.catalog.assets
            if asset.asset_id in candidates
        ]

    def _score_asset(
        self,
        asset: AssetEntry,
        category: str,
        description: str,
        attributes: dict[str, Any],
        est_dims: dict[str, float],
        dim_tolerance: float
    ) -> tuple[float, list[str]]:
        """Score an asset for a given object specification."""
        score = 0.0
        reasons = []

        # Category match
        if asset.category.lower() == category:
            score += 0.3
            reasons.append("category_exact_match")
        elif category in asset.category.lower() or asset.category.lower() in category:
            score += 0.15
            reasons.append("category_partial_match")

        # Tag overlap
        desc_words = set(self._tokenize(description))
        asset_tags = set(t.lower() for t in asset.tags)
        overlap = desc_words & asset_tags
        tag_score = min(len(overlap) * 0.1, 0.3)
        if tag_score > 0:
            score += tag_score
            reasons.append(f"tags_matched:{','.join(overlap)}")

        # Dimension matching
        if asset.dimensions and est_dims:
            dim_score = self._score_dimensions(
                asset.dimensions, est_dims, dim_tolerance
            )
            if dim_score > 0:
                score += dim_score
                reasons.append("dimensions_compatible")

        # Attribute matching
        if attributes:
            attr_score = self._score_attributes(asset, attributes)
            if attr_score > 0:
                score += attr_score
                reasons.append("attributes_matched")

        # SimReady bonus
        if asset.simready_metadata and asset.simready_metadata.get("is_simready"):
            score += 0.1
            reasons.append("simready")

        return score, reasons

    def _score_dimensions(
        self,
        asset_dims: dict[str, float],
        est_dims: dict[str, float],
        tolerance: float
    ) -> float:
        """Score dimension compatibility."""
        if not asset_dims or not est_dims:
            return 0.0

        matches = 0
        total = 0

        for dim in ["width", "depth", "height"]:
            if dim in asset_dims and dim in est_dims:
                asset_val = asset_dims[dim]
                est_val = est_dims[dim]

                if est_val > 0:
                    ratio = asset_val / est_val
                    if (1 - tolerance) <= ratio <= (1 + tolerance):
                        matches += 1
                total += 1

        if total == 0:
            return 0.0

        return (matches / total) * 0.2

    def _score_attributes(
        self,
        asset: AssetEntry,
        attributes: dict[str, Any]
    ) -> float:
        """Score attribute matching (material, color, style)."""
        score = 0.0

        # Material matching
        if "material" in attributes:
            material = attributes["material"].lower()
            if any(material in m.lower() for m in asset.materials):
                score += 0.05
            if any(material in t.lower() for t in asset.tags):
                score += 0.05

        # Color matching (from tags)
        if "color" in attributes:
            color = attributes["color"].lower()
            if any(color in t.lower() for t in asset.tags):
                score += 0.05

        # Style matching
        if "style" in attributes:
            style = attributes["style"].lower()
            if any(style in t.lower() for t in asset.tags):
                score += 0.05

        return min(score, 0.2)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into searchable words."""
        import re
        # Split on non-alphanumeric
        words = re.split(r'[^a-zA-Z0-9]+', text.lower())
        # Filter short words
        return [w for w in words if len(w) > 2]

    def _get_default_variants(self, asset: AssetEntry) -> dict[str, str]:
        """Get default variant selections for an asset."""
        variants = {}
        for vset in asset.variant_sets:
            if vset.get("default"):
                variants[vset["name"]] = vset["default"]
            elif vset.get("variants"):
                variants[vset["name"]] = vset["variants"][0]
        return variants

    def to_matched_assets(
        self,
        results: dict[str, MatchResult]
    ) -> dict[str, dict[str, Any]]:
        """Convert match results to format expected by RecipeCompiler."""
        matched = {}
        for obj_id, result in results.items():
            if result.chosen:
                matched[obj_id] = {
                    "chosen_path": result.chosen.asset_path,
                    "candidates": [
                        {
                            "asset_path": c.asset_path,
                            "score": c.score,
                            "dimensions": c.dimensions
                        }
                        for c in result.candidates
                    ],
                    "variants": result.chosen.variants or {},
                    "pack_name": self.catalog.pack_name
                }
            else:
                matched[obj_id] = {
                    "chosen_path": "",
                    "candidates": [
                        {
                            "asset_path": c.asset_path,
                            "score": c.score,
                            "dimensions": c.dimensions
                        }
                        for c in result.candidates
                    ],
                    "variants": {},
                    "pack_name": self.catalog.pack_name,
                    "needs_selection": True
                }
        return matched
