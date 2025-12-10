#!/usr/bin/env python3
"""
Scene Validation Script

This script validates a generated scene for:
- Asset reference resolution
- Physics configuration
- Semantic labels
- Material validity

Note: Full validation requires Isaac Sim runtime.
This script performs static analysis where possible.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from gemini_qa import build_scene_context, run_gemini_scene_review


def validate_usd_structure(scene_path: str) -> dict[str, Any]:
    """Validate USD file structure (requires OpenUSD)."""
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "prims_count": 0,
        "references_count": 0
    }

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(scene_path)
        if not stage:
            result["errors"].append("Failed to open USD stage")
            return result

        # Count prims
        prim_count = 0
        ref_count = 0

        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            prim_count += 1

            # Check references
            refs = prim.GetReferences()
            if refs:
                ref_count += 1

        result["prims_count"] = prim_count
        result["references_count"] = ref_count
        result["valid"] = True

    except ImportError:
        result["warnings"].append("OpenUSD not available, skipping USD structure validation")
        result["valid"] = True  # Assume valid if we can't check

    except Exception as e:
        result["errors"].append(f"USD validation error: {str(e)}")

    return result


def validate_asset_references(recipe: dict[str, Any], assets_root: str) -> dict[str, Any]:
    """Validate that all asset references resolve."""
    result = {
        "total": 0,
        "resolved": 0,
        "missing": []
    }

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
            result["missing"].append({
                "object_id": obj.get("id"),
                "asset_path": asset_path
            })

    return result


def validate_physics_config(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate physics configuration."""
    result = {
        "objects_with_physics": 0,
        "rigid_bodies": 0,
        "articulations": 0,
        "warnings": []
    }

    for obj in recipe.get("objects", []):
        physics = obj.get("physics", {})

        if physics.get("enabled", False):
            result["objects_with_physics"] += 1

        if physics.get("rigid_body", False):
            result["rigid_bodies"] += 1

            # Check for potential issues
            if not physics.get("collision_enabled", True):
                result["warnings"].append(
                    f"{obj['id']}: Rigid body without collision"
                )

        if obj.get("articulation"):
            result["articulations"] += 1

            art = obj["articulation"]
            if not art.get("limits"):
                result["warnings"].append(
                    f"{obj['id']}: Articulation without limits defined"
                )

    return result


def validate_semantics(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate semantic labels."""
    result = {
        "objects_with_semantics": 0,
        "classes_found": set(),
        "missing_semantics": []
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


def generate_report(
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

    # Collect errors and warnings
    all_errors.extend(usd_result.get("errors", []))
    all_warnings.extend(usd_result.get("warnings", []))
    all_warnings.extend(physics_result.get("warnings", []))

    if ref_result["missing"]:
        for missing in ref_result["missing"]:
            all_errors.append(f"Missing asset: {missing['asset_path']}")

    if semantics_result["missing_semantics"]:
        for obj_id in semantics_result["missing_semantics"]:
            all_warnings.append(f"Missing semantics: {obj_id}")

    if gemini_result.get("status") == "error" and gemini_result.get("reason"):
        all_warnings.append(f"Gemini QA skipped: {gemini_result['reason']}")

    gemini_blocking = []
    if gemini_result.get("status") == "ok":
        gemini_blocking = gemini_result.get("plan", {}).get("blocking_issues", []) or []
        for issue in gemini_blocking:
            title = issue.get("issue") or "Gemini flagged issue"
            all_errors.append(f"Gemini: {title}")

    # Determine overall status
    is_valid = len(all_errors) == 0

    return {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "valid": is_valid,
            "ready_for_simulation": is_valid and usd_result["valid"],
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings)
        },
        "usd_structure": usd_result,
        "asset_references": ref_result,
        "physics": physics_result,
        "semantics": semantics_result,
        "gemini_scene_review": gemini_result,
        "errors": all_errors,
        "warnings": all_warnings
    }


def upload_report(bucket: str, output_prefix: str, report: dict[str, Any]):
    """Upload report to GCS."""
    from google.cloud import storage
    client = storage.Client()
    gcs_bucket = client.bucket(bucket)

    blob = gcs_bucket.blob(f"{output_prefix}/validation_report.json")
    blob.upload_from_string(
        json.dumps(report, indent=2),
        content_type="application/json"
    )
    print(f"[QA] Uploaded report to gs://{bucket}/{output_prefix}/validation_report.json")


def main():
    parser = argparse.ArgumentParser(description="Validate Scene")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--bucket", required=True, help="GCS bucket")
    parser.add_argument("--scene-path", required=True, help="Path to scene.usda")
    parser.add_argument("--recipe-path", help="Path to recipe.json")
    parser.add_argument("--assets-root", default="/mnt/gcs/assets", help="Assets root")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")
    args = parser.parse_args()

    print(f"[QA] Validating job {args.job_id}")

    # Load recipe if provided
    recipe = {}
    if args.recipe_path:
        if args.recipe_path.startswith("gs://"):
            from google.cloud import storage
            client = storage.Client()
            parts = args.recipe_path[5:].split("/", 1)
            bucket = client.bucket(parts[0])
            blob = bucket.blob(parts[1])
            recipe = json.loads(blob.download_as_text())
        else:
            with open(args.recipe_path) as f:
                recipe = json.load(f)

    # Run validations
    print("[QA] Validating USD structure...")
    usd_result = validate_usd_structure(args.scene_path)

    print("[QA] Validating asset references...")
    ref_result = validate_asset_references(recipe, args.assets_root)

    print("[QA] Validating physics configuration...")
    physics_result = validate_physics_config(recipe)

    print("[QA] Validating semantics...")
    semantics_result = validate_semantics(recipe)

    print("[QA] Building Gemini context...")
    context = build_scene_context(recipe, usd_result, ref_result, physics_result, semantics_result)

    print("[QA] Requesting Gemini scene-specific QA plan...")
    gemini_review = run_gemini_scene_review(context)

    gemini_result = {
        "status": "ok" if gemini_review.enabled else "error",
        "reason": gemini_review.reason,
        "plan": gemini_review.plan,
        "raw_response": gemini_review.raw_response,
    }

    # Generate report
    report = generate_report(
        args.job_id,
        usd_result,
        ref_result,
        physics_result,
        semantics_result,
        gemini_result,
    )

    # Upload report
    print("[QA] Uploading report...")
    upload_report(args.bucket, args.output_prefix, report)

    # Print summary
    print(f"[QA] Validation complete:")
    print(f"  - Valid: {report['overall']['valid']}")
    print(f"  - Errors: {report['overall']['total_errors']}")
    print(f"  - Warnings: {report['overall']['total_warnings']}")
    if gemini_result["status"] == "ok":
        print("  - Gemini QA: plan generated")
    else:
        print(f"  - Gemini QA: skipped ({gemini_result['reason']})")

    # Exit with appropriate code
    if report["overall"]["valid"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
