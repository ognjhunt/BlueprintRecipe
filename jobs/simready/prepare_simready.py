#!/usr/bin/env python3
"""
SimReady Preparation Script

This script prepares assets and validates the recipe for Isaac Sim/Lab.
It runs as a Cloud Run Job and handles:
- Recipe validation
- Asset path resolution
- Basic physics checks
- Output generation
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def load_recipe(recipe_path: str) -> dict[str, Any]:
    """Load recipe from GCS or local path."""
    if recipe_path.startswith("gs://"):
        from google.cloud import storage
        client = storage.Client()

        # Parse GCS path
        parts = recipe_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        return json.loads(content)
    else:
        with open(recipe_path) as f:
            return json.load(f)


def validate_recipe(recipe: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Validate recipe structure and content."""
    errors = []
    warnings = []

    # Check required fields
    if "version" not in recipe:
        errors.append("Missing version field")

    if "metadata" not in recipe:
        errors.append("Missing metadata field")

    if "objects" not in recipe:
        errors.append("Missing objects field")
    elif not recipe["objects"]:
        warnings.append("No objects in recipe")

    # Check object definitions
    for obj in recipe.get("objects", []):
        obj_id = obj.get("id", "unknown")

        if not obj.get("chosen_asset", {}).get("asset_path"):
            errors.append(f"Object {obj_id} has no chosen asset path")

        if not obj.get("transform"):
            warnings.append(f"Object {obj_id} has no transform")

        if not obj.get("semantics", {}).get("class"):
            warnings.append(f"Object {obj_id} has no semantic class")

    # Check asset packs
    if not recipe.get("asset_packs"):
        warnings.append("No asset packs specified")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def resolve_asset_paths(
    recipe: dict[str, Any],
    assets_root: str
) -> tuple[dict[str, str], list[str]]:
    """Resolve and validate asset paths."""
    resolved = {}
    missing = []

    assets_path = Path(assets_root)

    for obj in recipe.get("objects", []):
        obj_id = obj.get("id", "unknown")
        asset_path = obj.get("chosen_asset", {}).get("asset_path", "")

        if not asset_path:
            continue

        # Try to resolve the path
        full_path = assets_path / asset_path

        if full_path.exists():
            resolved[obj_id] = str(full_path)
        else:
            # Try alternative paths
            alt_paths = [
                assets_path / asset_path.replace("/", os.sep),
                assets_path / Path(asset_path).name,
            ]

            found = False
            for alt in alt_paths:
                if alt.exists():
                    resolved[obj_id] = str(alt)
                    found = True
                    break

            if not found:
                missing.append(f"{obj_id}: {asset_path}")

    return resolved, missing


def check_physics_config(recipe: dict[str, Any]) -> list[str]:
    """Check physics configuration for common issues."""
    warnings = []

    for obj in recipe.get("objects", []):
        obj_id = obj.get("id", "unknown")
        physics = obj.get("physics", {})

        # Check for missing collision config on rigid bodies
        if physics.get("rigid_body") and not physics.get("collision_enabled", True):
            warnings.append(f"{obj_id}: Rigid body without collision")

        # Check for articulated objects
        articulation = obj.get("articulation")
        if articulation:
            if not articulation.get("limits"):
                warnings.append(f"{obj_id}: Articulation without limits")

    return warnings


def generate_qa_report(
    recipe: dict[str, Any],
    validation_result: tuple[bool, list[str], list[str]],
    resolved_paths: dict[str, str],
    missing_assets: list[str],
    physics_warnings: list[str]
) -> dict[str, Any]:
    """Generate QA report."""
    is_valid, errors, warnings = validation_result

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "recipe_version": recipe.get("version", "unknown"),
        "environment_type": recipe.get("metadata", {}).get("environment_type", "unknown"),
        "validation": {
            "is_valid": is_valid and len(missing_assets) == 0,
            "errors": errors,
            "warnings": warnings
        },
        "asset_resolution": {
            "total_objects": len(recipe.get("objects", [])),
            "resolved": len(resolved_paths),
            "missing": missing_assets
        },
        "physics": {
            "warnings": physics_warnings
        },
        "summary": {
            "ready_for_simulation": is_valid and len(missing_assets) == 0,
            "total_errors": len(errors) + len(missing_assets),
            "total_warnings": len(warnings) + len(physics_warnings)
        }
    }


def upload_results(
    bucket: str,
    output_prefix: str,
    qa_report: dict[str, Any],
    resolved_paths: dict[str, str]
):
    """Upload results to GCS."""
    from google.cloud import storage
    client = storage.Client()
    gcs_bucket = client.bucket(bucket)

    # Upload QA report
    qa_blob = gcs_bucket.blob(f"{output_prefix}/qa_report.json")
    qa_blob.upload_from_string(
        json.dumps(qa_report, indent=2),
        content_type="application/json"
    )
    print(f"[SIMREADY] Uploaded QA report to gs://{bucket}/{output_prefix}/qa_report.json")

    # Upload resolved paths
    paths_blob = gcs_bucket.blob(f"{output_prefix}/resolved_paths.json")
    paths_blob.upload_from_string(
        json.dumps(resolved_paths, indent=2),
        content_type="application/json"
    )
    print(f"[SIMREADY] Uploaded resolved paths to gs://{bucket}/{output_prefix}/resolved_paths.json")


def main():
    parser = argparse.ArgumentParser(description="SimReady Preparation")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--bucket", required=True, help="GCS bucket")
    parser.add_argument("--recipe-path", required=True, help="Path to recipe.json")
    parser.add_argument("--assets-root", default="/mnt/gcs/assets", help="Assets root path")
    parser.add_argument("--output-prefix", required=True, help="Output prefix in bucket")
    args = parser.parse_args()

    print(f"[SIMREADY] Processing job {args.job_id}")

    # Load recipe
    print(f"[SIMREADY] Loading recipe from {args.recipe_path}")
    recipe = load_recipe(args.recipe_path)

    # Validate recipe
    print("[SIMREADY] Validating recipe...")
    validation_result = validate_recipe(recipe)
    is_valid, errors, warnings = validation_result

    if errors:
        print(f"[SIMREADY] Validation errors: {errors}")
    if warnings:
        print(f"[SIMREADY] Validation warnings: {warnings}")

    # Resolve asset paths
    print(f"[SIMREADY] Resolving asset paths from {args.assets_root}")
    resolved_paths, missing_assets = resolve_asset_paths(recipe, args.assets_root)

    print(f"[SIMREADY] Resolved {len(resolved_paths)} assets")
    if missing_assets:
        print(f"[SIMREADY] Missing assets: {missing_assets}")

    # Check physics configuration
    print("[SIMREADY] Checking physics configuration...")
    physics_warnings = check_physics_config(recipe)
    if physics_warnings:
        print(f"[SIMREADY] Physics warnings: {physics_warnings}")

    # Generate QA report
    qa_report = generate_qa_report(
        recipe,
        validation_result,
        resolved_paths,
        missing_assets,
        physics_warnings
    )

    # Upload results
    print("[SIMREADY] Uploading results...")
    upload_results(args.bucket, args.output_prefix, qa_report, resolved_paths)

    # Exit with appropriate code
    if qa_report["summary"]["ready_for_simulation"]:
        print("[SIMREADY] Recipe is ready for simulation")
        sys.exit(0)
    else:
        print("[SIMREADY] Recipe has issues that need resolution")
        sys.exit(1)


if __name__ == "__main__":
    main()
