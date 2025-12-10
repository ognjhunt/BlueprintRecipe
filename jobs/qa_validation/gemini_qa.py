"""Gemini-powered scene QA review utilities.

This module prepares rich, scene-specific context and asks Gemini 3.0 Pro to
propose validation focus areas and test cases tailored to the current recipe and
static validation results. The prompts are verbose by design to give Gemini all
available evidence while keeping the output constrained to JSON for
machine-consumption.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import importlib.util


@dataclass
class GeminiQAResult:
    """Container for Gemini QA output."""

    enabled: bool
    reason: str
    raw_response: Optional[str]
    plan: Dict[str, Any]


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return cleaned


def _safe_truncate_list(items: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    if len(items) <= limit:
        return items
    return items[:limit]


def build_scene_context(
    recipe: Dict[str, Any],
    usd_result: Dict[str, Any],
    ref_result: Dict[str, Any],
    physics_result: Dict[str, Any],
    semantics_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize the scene into a compact context for Gemini."""

    objects = recipe.get("objects", []) if isinstance(recipe, dict) else []
    categories: Dict[str, int] = {}
    semantic_classes: Dict[str, int] = {}
    dynamic_objects: List[str] = []
    articulation_objects: List[str] = []

    sample_objects: List[Dict[str, Any]] = []
    for obj in objects:
        category = (obj.get("category") or obj.get("class") or "unknown").lower()
        categories[category] = categories.get(category, 0) + 1

        semantics = obj.get("semantics", {})
        sem_class = (semantics.get("class") or semantics.get("label") or "unknown").lower()
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

    sample_objects = _safe_truncate_list(sample_objects)

    context = {
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

    return context


def build_gemini_prompt(context: Dict[str, Any]) -> str:
    """Create a detailed prompt for Gemini 3.0 Pro QA review."""

    prompt = f"""
You are the QA lead for a robotics simulation pipeline. You must design scene-specific
validation and testing for USD assets destined for NVIDIA Isaac Sim and Replicator.
Use the provided context to produce a concrete, *scene-tailored* plan.
Avoid generic guidance; every recommendation should reference the observed scene facts.

Context (JSON):
{json.dumps(context, indent=2)}

Instructions:
- Base your plan on the exact object ids, categories, and validation results above.
- Recommend additional checks only when they apply to the current scene; do not
  include placeholders for objects that do not exist here.
- If you suggest physics or articulation tests, align them with the reported
  masses, friction, collision shapes, and articulation flags. Prefer per-object
  checks (e.g., "fridge door articulation on obj_12").
- Prioritize high-risk items: missing assets, unresolved references, missing
  semantics, or heavy/dynamic objects without clear collision setup.
- Provide measurable expectations (what success/failure looks like) so the QA
  runner can act on the output automatically.

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
    "render": [ ... same structure ... ],
    "semantics": [ ... same structure ... ],
    "articulation": [ ... same structure ... ]
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
"""
    return prompt.strip()


def _get_genai_modules():
    """Load google-genai modules if available."""

    spec_client = importlib.util.find_spec("google.genai")
    spec_types = importlib.util.find_spec("google.genai.types")
    if spec_client is None or spec_types is None:
        return None, None

    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    return genai, types


def _create_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    genai, _ = _get_genai_modules()
    if genai is None:
        return None

    return genai.Client(api_key=api_key)


def run_gemini_scene_review(context: Dict[str, Any]) -> GeminiQAResult:
    """Run Gemini 3.0 Pro to generate a scene-specific QA plan."""

    client = _create_genai_client()
    if client is None:
        return GeminiQAResult(
            enabled=False,
            reason="Gemini unavailable (missing API key or google-genai)",
            raw_response=None,
            plan={},
        )

    _, types = _get_genai_modules()
    if types is None:
        return GeminiQAResult(
            enabled=False,
            reason="google-genai types unavailable",
            raw_response=None,
            plan={},
        )

    prompt = build_gemini_prompt(context)
    model = os.getenv("GEMINI_MODEL", "gemini-3.0-pro")

    try:
        cfg = types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )
        raw = _strip_code_fences(response.text or "")
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("Gemini returned non-object JSON")

        return GeminiQAResult(
            enabled=True,
            reason="Gemini response received",
            raw_response=raw,
            plan=plan,
        )

    except Exception as exc:  # pragma: no cover - network/SDK failures
        return GeminiQAResult(
            enabled=False,
            reason=f"Gemini call failed: {exc}",
            raw_response=None,
            plan={},
        )
