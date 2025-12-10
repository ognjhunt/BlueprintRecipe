"""Scene validation service for BlueprintRecipe API.

This module provides async-friendly validation functions that can be called
inline during the pipeline processing. It wraps the static validation logic
from jobs/qa_validation/ and integrates with Gemini for scene review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""

    valid: bool
    ready_for_simulation: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usd_structure: dict[str, Any] = field(default_factory=dict)
    asset_references: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    semantics: dict[str, Any] = field(default_factory=dict)
    gemini_review: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


def validate_usd_structure(scene_path: str | None) -> dict[str, Any]:
    """Validate USD file structure (requires OpenUSD)."""
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "prims_count": 0,
        "references_count": 0,
    }

    if not scene_path:
        result["warnings"].append("No scene path provided, skipping USD validation")
        result["valid"] = True
        return result

    scene_file = Path(scene_path)
    if not scene_file.exists():
        result["warnings"].append(f"Scene file not found: {scene_path}")
        result["valid"] = True  # Allow pipeline to continue
        return result

    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(scene_file))
        if not stage:
            result["errors"].append("Failed to open USD stage")
            return result

        prim_count = 0
        ref_count = 0

        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            prim_count += 1
            refs = prim.GetReferences()
            if refs:
                ref_count += 1

        result["prims_count"] = prim_count
        result["references_count"] = ref_count
        result["valid"] = True

    except ImportError:
        result["warnings"].append(
            "OpenUSD not available, skipping USD structure validation"
        )
        result["valid"] = True

    except Exception as e:
        result["errors"].append(f"USD validation error: {str(e)}")

    return result


def validate_asset_references(
    recipe: dict[str, Any], assets_root: str | None
) -> dict[str, Any]:
    """Validate that all asset references resolve."""
    result = {"total": 0, "resolved": 0, "missing": [], "skipped": False}

    if not assets_root:
        result["skipped"] = True
        return result

    assets_path = Path(assets_root)

    for obj in recipe.get("objects", []):
        asset_path = obj.get("chosen_asset", {}).get("asset_path", "")
        if not asset_path:
            continue

        result["total"] += 1
        full_path = assets_path / asset_path

        if full_path.exists():
            result["resolved"] += 1
        else:
            result["missing"].append(
                {"object_id": obj.get("id"), "asset_path": asset_path}
            )

    return result


def validate_physics_config(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate physics configuration."""
    result = {
        "objects_with_physics": 0,
        "rigid_bodies": 0,
        "articulations": 0,
        "warnings": [],
    }

    for obj in recipe.get("objects", []):
        physics = obj.get("physics", {})

        if physics.get("enabled", False):
            result["objects_with_physics"] += 1

        if physics.get("rigid_body", False):
            result["rigid_bodies"] += 1

            if not physics.get("collision_enabled", True):
                result["warnings"].append(
                    f"{obj.get('id', 'unknown')}: Rigid body without collision"
                )

        if obj.get("articulation"):
            result["articulations"] += 1

            art = obj["articulation"]
            if not art.get("limits"):
                result["warnings"].append(
                    f"{obj.get('id', 'unknown')}: Articulation without limits defined"
                )

    return result


def validate_semantics(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate semantic labels."""
    result = {
        "objects_with_semantics": 0,
        "classes_found": set(),
        "missing_semantics": [],
    }

    for obj in recipe.get("objects", []):
        semantics = obj.get("semantics", {})

        if semantics.get("class"):
            result["objects_with_semantics"] += 1
            result["classes_found"].add(semantics["class"])
        else:
            result["missing_semantics"].append(obj.get("id"))

    result["classes_found"] = list(result["classes_found"])
    return result


def _build_scene_context(
    recipe: dict[str, Any],
    usd_result: dict[str, Any],
    ref_result: dict[str, Any],
    physics_result: dict[str, Any],
    semantics_result: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the scene into a compact context for Gemini."""
    objects = recipe.get("objects", []) if isinstance(recipe, dict) else []
    categories: dict[str, int] = {}
    semantic_classes: dict[str, int] = {}
    dynamic_objects: list[str] = []
    articulation_objects: list[str] = []

    sample_objects: list[dict[str, Any]] = []
    for obj in objects[:12]:  # Limit to first 12 for context
        category = (obj.get("category") or obj.get("class") or "unknown").lower()
        categories[category] = categories.get(category, 0) + 1

        semantics = obj.get("semantics", {})
        sem_class = (
            semantics.get("class") or semantics.get("label") or "unknown"
        ).lower()
        semantic_classes[sem_class] = semantic_classes.get(sem_class, 0) + 1

        physics = obj.get("physics", {})
        if physics.get("enabled"):
            dynamic_objects.append(str(obj.get("id")))
        if obj.get("articulation"):
            articulation_objects.append(str(obj.get("id")))

        sample_objects.append(
            {
                "id": obj.get("id"),
                "name": obj.get("name"),
                "category": obj.get("category"),
                "semantic_class": semantics.get("class"),
                "asset_path": obj.get("chosen_asset", {}).get("asset_path"),
                "mass": physics.get("mass") or physics.get("mass_kg"),
                "friction": {
                    "static": physics.get("static_friction"),
                    "dynamic": physics.get("dynamic_friction"),
                },
                "collision": physics.get("collision_shape") or physics.get("collision"),
                "articulation": bool(obj.get("articulation")),
            }
        )

    return {
        "scene_stats": {
            "object_count": len(objects),
            "category_histogram": categories,
            "semantic_histogram": semantic_classes,
            "dynamic_object_ids": dynamic_objects,
            "articulation_object_ids": articulation_objects,
        },
        "static_validation": {
            "usd": usd_result,
            "asset_references": ref_result,
            "physics": physics_result,
            "semantics": semantics_result,
        },
        "sample_objects": sample_objects,
    }


def _run_gemini_review(context: dict[str, Any]) -> dict[str, Any]:
    """Run Gemini scene review if available."""
    result = {"status": "skipped", "reason": "", "plan": {}, "raw_response": None}

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        result["reason"] = "No API key configured (GOOGLE_API_KEY or GEMINI_API_KEY)"
        return result

    try:
        import importlib.util

        if importlib.util.find_spec("google.genai") is None:
            result["reason"] = "google-genai package not installed"
            return result

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = _build_gemini_prompt(context)
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        cfg = types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )

        raw = (response.text or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("Gemini returned non-object JSON")

        result["status"] = "ok"
        result["reason"] = "Gemini response received"
        result["raw_response"] = raw
        result["plan"] = plan

    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"Gemini call failed: {exc}"
        logger.warning("Gemini QA review failed: %s", exc)

    return result


def _build_gemini_prompt(context: dict[str, Any]) -> str:
    """Create a prompt for Gemini QA review."""
    return f"""
You are the QA lead for a robotics simulation pipeline. You must design scene-specific
validation and testing for USD assets destined for NVIDIA Isaac Sim and Replicator.
Use the provided context to produce a concrete, *scene-tailored* plan.
Avoid generic guidance; every recommendation should reference the observed scene facts.

Context (JSON):
{json.dumps(context, indent=2)}

Instructions:
- Base your plan on the exact object ids, categories, and validation results above.
- Recommend additional checks only when they apply to the current scene.
- If you suggest physics or articulation tests, align them with the reported
  masses, friction, collision shapes, and articulation flags.
- Prioritize high-risk items: missing assets, unresolved references, missing
  semantics, or heavy/dynamic objects without clear collision setup.
- Provide measurable expectations (what success/failure looks like).

Respond ONLY with JSON using this schema:
{{
  "blocking_issues": [
    {{"issue": "<short description>", "evidence": "<why this matters>", "object_ids": ["<id>", ...]}}
  ],
  "targeted_tests": {{
    "physics": [
      {{
        "name": "<test name>",
        "objective": "<what to verify>",
        "steps": ["<step1>", "<step2>", ...],
        "expected": "<pass criteria>",
        "object_ids": ["<id>", ...],
        "failure_signals": ["<observable symptom>", ...]
      }}
    ],
    "render": [],
    "semantics": [],
    "articulation": []
  }},
  "metrics": [
    {{"name": "<metric name>", "definition": "<how to compute>", "targets": "<which objects/prims>"}}
  ],
  "recommended_autofixes": [
    {{"description": "<what to change>", "scope": "<usd|recipe|assets>", "object_ids": ["<id>"]}}
  ],
  "notes": "<concise scene-aware summary>"
}}

The JSON must be valid and reflect only the current scene.
""".strip()


def _generate_report(
    job_id: str,
    usd_result: dict[str, Any],
    ref_result: dict[str, Any],
    physics_result: dict[str, Any],
    semantics_result: dict[str, Any],
    gemini_result: dict[str, Any],
) -> dict[str, Any]:
    """Generate comprehensive QA report."""
    all_errors = []
    all_warnings = []

    all_errors.extend(usd_result.get("errors", []))
    all_warnings.extend(usd_result.get("warnings", []))
    all_warnings.extend(physics_result.get("warnings", []))

    if ref_result.get("missing"):
        for missing in ref_result["missing"]:
            all_errors.append(f"Missing asset: {missing.get('asset_path', 'unknown')}")

    if semantics_result.get("missing_semantics"):
        for obj_id in semantics_result["missing_semantics"]:
            all_warnings.append(f"Missing semantics: {obj_id}")

    if gemini_result.get("status") == "error" and gemini_result.get("reason"):
        all_warnings.append(f"Gemini QA skipped: {gemini_result['reason']}")
    elif gemini_result.get("status") == "skipped" and gemini_result.get("reason"):
        all_warnings.append(f"Gemini QA skipped: {gemini_result['reason']}")

    gemini_blocking = []
    if gemini_result.get("status") == "ok":
        gemini_blocking = gemini_result.get("plan", {}).get("blocking_issues", []) or []
        for issue in gemini_blocking:
            title = issue.get("issue") or "Gemini flagged issue"
            all_errors.append(f"Gemini: {title}")

    is_valid = len(all_errors) == 0

    return {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "valid": is_valid,
            "ready_for_simulation": is_valid and usd_result.get("valid", False),
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
        },
        "usd_structure": usd_result,
        "asset_references": ref_result,
        "physics": physics_result,
        "semantics": semantics_result,
        "gemini_scene_review": gemini_result,
        "errors": all_errors,
        "warnings": all_warnings,
    }


async def validate_scene(
    job_id: str,
    recipe: dict[str, Any],
    scene_path: str | None = None,
    assets_root: str | None = None,
    skip_gemini: bool = False,
) -> ValidationResult:
    """
    Run full scene validation asynchronously.

    Args:
        job_id: The job identifier for reporting.
        recipe: The compiled recipe dictionary.
        scene_path: Optional path to the scene USD file.
        assets_root: Optional root path for asset resolution.
        skip_gemini: If True, skip Gemini AI review.

    Returns:
        ValidationResult with all validation findings.
    """
    logger.info("[Validation] Starting validation for job %s", job_id)

    # Run CPU-bound validations in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    usd_result = await loop.run_in_executor(None, validate_usd_structure, scene_path)
    logger.debug("[Validation] USD structure: %s", usd_result)

    ref_result = await loop.run_in_executor(
        None, validate_asset_references, recipe, assets_root
    )
    logger.debug("[Validation] Asset references: %s", ref_result)

    physics_result = await loop.run_in_executor(None, validate_physics_config, recipe)
    logger.debug("[Validation] Physics config: %s", physics_result)

    semantics_result = await loop.run_in_executor(None, validate_semantics, recipe)
    logger.debug("[Validation] Semantics: %s", semantics_result)

    # Build context for Gemini
    context = _build_scene_context(
        recipe, usd_result, ref_result, physics_result, semantics_result
    )

    # Run Gemini review (network I/O, but SDK is sync)
    if skip_gemini:
        gemini_result = {
            "status": "skipped",
            "reason": "Gemini review disabled",
            "plan": {},
            "raw_response": None,
        }
    else:
        gemini_result = await loop.run_in_executor(None, _run_gemini_review, context)

    logger.debug("[Validation] Gemini review status: %s", gemini_result.get("status"))

    # Generate report
    report = _generate_report(
        job_id, usd_result, ref_result, physics_result, semantics_result, gemini_result
    )

    logger.info(
        "[Validation] Complete - valid=%s, errors=%d, warnings=%d",
        report["overall"]["valid"],
        report["overall"]["total_errors"],
        report["overall"]["total_warnings"],
    )

    return ValidationResult(
        valid=report["overall"]["valid"],
        ready_for_simulation=report["overall"]["ready_for_simulation"],
        errors=report.get("errors", []),
        warnings=report.get("warnings", []),
        usd_structure=usd_result,
        asset_references=ref_result,
        physics=physics_result,
        semantics=semantics_result,
        gemini_review=gemini_result,
        report=report,
    )
