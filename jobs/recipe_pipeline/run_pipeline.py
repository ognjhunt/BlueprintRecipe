#!/usr/bin/env python3
"""
BlueprintRecipe Pipeline Job

Cloud Run Job that processes an uploaded image through the full pipeline:
1. Scene Planning (Gemini vision analysis)
2. Asset Matching (catalog + embeddings)
3. Recipe Compilation (USD + physics + semantics)
4. Validation (structure + Gemini QA)

Triggered by Eventarc when an image is uploaded to:
  gs://{bucket}/scenes/{scene_id}/images/{filename}

Outputs are written to:
  gs://{bucket}/scenes/{scene_id}/recipe/
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directories to path for imports
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")

from src.asset_catalog.vector_store import VectorStoreClient, VectorStoreConfig


def _get_float_env(var_name: str, default: float) -> float:
    """Return float value from environment with a safe fallback."""

    try:
        return float(os.getenv(var_name, default))
    except (TypeError, ValueError):
        print(
            f"[PIPELINE] Warning: Invalid value for {var_name}; using default {default}"
        )
        return default


def _build_vector_store_from_env() -> Optional[VectorStoreClient]:
    """Create a vector store client when configuration is provided."""

    provider = os.getenv("VECTOR_STORE_PROVIDER")
    if not provider:
        return None

    try:
        config = VectorStoreConfig(
            provider=provider,
            collection=os.getenv("VECTOR_STORE_COLLECTION", "asset-embeddings"),
            connection_uri=os.getenv("VECTOR_STORE_URI"),
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        )
        return VectorStoreClient(config)
    except Exception as exc:  # pragma: no cover - runtime configuration
        print(f"[PIPELINE] Warning: failed to initialize vector store: {exc}")
        return None


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Download a file from GCS to local path."""
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(local_path)
    print(f"[PIPELINE] Downloaded {gcs_uri} to {local_path}")
    return local_path


def upload_to_gcs(local_path: str, gcs_uri: str, content_type: str = "application/json") -> str:
    """Upload a file to GCS."""
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path, content_type=content_type)
    print(f"[PIPELINE] Uploaded {local_path} to {gcs_uri}")
    return gcs_uri


def upload_string_to_gcs(content: str, gcs_uri: str, content_type: str = "application/json") -> str:
    """Upload a string to GCS."""
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(content, content_type=content_type)
    print(f"[PIPELINE] Uploaded content to {gcs_uri}")
    return gcs_uri


def upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> list[str]:
    """Upload all files in a directory to GCS."""
    from google.cloud import storage

    parts = gcs_prefix[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = []
    local_path = Path(local_dir)

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_path)
            blob_name = f"{prefix}/{relative}".replace("\\", "/")
            blob = bucket.blob(blob_name)

            # Determine content type
            suffix = file_path.suffix.lower()
            content_types = {
                ".json": "application/json",
                ".yaml": "application/x-yaml",
                ".yml": "application/x-yaml",
                ".usd": "application/octet-stream",
                ".usda": "text/plain",
                ".usdc": "application/octet-stream",
                ".py": "text/x-python",
            }
            content_type = content_types.get(suffix, "application/octet-stream")

            blob.upload_from_filename(str(file_path), content_type=content_type)
            uploaded.append(f"gs://{bucket_name}/{blob_name}")

    print(f"[PIPELINE] Uploaded {len(uploaded)} files to {gcs_prefix}")
    return uploaded


def extract_scene_id(gcs_uri: str) -> str:
    """Extract scene_id from GCS path like gs://bucket/scenes/{scene_id}/images/file.jpg"""
    # Pattern: gs://bucket/scenes/{scene_id}/images/{filename}
    path_parts = gcs_uri.replace("gs://", "").split("/")

    try:
        scenes_idx = path_parts.index("scenes")
        if scenes_idx + 1 < len(path_parts):
            return path_parts[scenes_idx + 1]
    except ValueError:
        pass

    # Fallback: use a hash of the path
    import hashlib
    return hashlib.sha1(gcs_uri.encode()).hexdigest()[:12]


def update_firestore_status(
    scene_id: str,
    status: str,
    phase: str = "",
    error: str = "",
    artifacts: dict = None
):
    """Update job status in Firestore."""
    try:
        from google.cloud import firestore

        project_id = os.getenv("FIRESTORE_PROJECT_ID", "blueprint-8c1ca")
        db = firestore.Client(project=project_id)

        doc_ref = db.collection("recipe_jobs").document(scene_id)

        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        if phase:
            update_data["phase"] = phase
        if error:
            update_data["error"] = error
        if artifacts:
            update_data["artifacts"] = artifacts

        doc_ref.set(update_data, merge=True)
        print(f"[PIPELINE] Updated Firestore: scene={scene_id}, status={status}, phase={phase}")

    except Exception as e:
        print(f"[PIPELINE] Warning: Failed to update Firestore: {e}")


def run_pipeline(
    image_uri: str,
    scene_id: str,
    bucket: str,
    environment_type: Optional[str] = None,
    task_intent: Optional[str] = None,
    target_policies: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run the full BlueprintRecipe pipeline."""

    from src.planning import ScenePlanner
    from src.asset_catalog import AssetCatalogBuilder, AssetMatcher, AssetEmbeddings
    from src.recipe_compiler import RecipeCompiler
    from src.recipe_compiler.compiler import CompilerConfig

    result = {
        "success": False,
        "scene_id": scene_id,
        "source_image_uri": image_uri,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "phases": {},
        "artifacts": {},
        "errors": [],
        "warnings": [],
    }

    output_prefix = f"scenes/{scene_id}/recipe"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_dir = tmpdir_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ============================================================
            # PHASE 1: Download image and plan scene
            # ============================================================
            update_firestore_status(scene_id, "processing", "planning")
            print("[PIPELINE] Phase 1: Scene Planning")

            # Download source image
            image_ext = Path(image_uri).suffix or ".jpg"
            local_image = str(tmpdir_path / f"source_image{image_ext}")
            download_from_gcs(image_uri, local_image)

            # Run scene planner
            planner = ScenePlanner()
            planning_result = planner.plan_from_image(
                image_path=local_image,
                task_intent=task_intent,
                environment_hint=environment_type,
                target_policies=target_policies or ["pick_place"]
            )

            if not planning_result.success or not planning_result.scene_plan:
                raise RuntimeError(f"Scene planning failed: {planning_result.error}")

            scene_plan = planning_result.scene_plan
            result["phases"]["planning"] = {
                "success": True,
                "objects_detected": len(scene_plan.get("object_inventory", [])),
                "environment_type": scene_plan.get("environment_type"),
            }

            if planning_result.warnings:
                result["warnings"].extend(planning_result.warnings)

            # Save scene plan
            scene_plan_path = output_dir / "scene_plan.json"
            with open(scene_plan_path, "w") as f:
                json.dump(scene_plan, f, indent=2)

            print(f"[PIPELINE] Scene plan: {len(scene_plan.get('object_inventory', []))} objects detected")

            # ============================================================
            # PHASE 2: Asset Matching
            # ============================================================
            update_firestore_status(scene_id, "processing", "matching")
            print("[PIPELINE] Phase 2: Asset Matching")

            # Load asset catalog
            catalog_path = Path("/app/asset_index.json")
            if not catalog_path.exists():
                catalog_path = Path(__file__).resolve().parents[2] / "asset_index.json"

            asset_catalog = AssetCatalogBuilder.load(str(catalog_path))

            # Load optional precomputed embeddings
            embeddings_db = None
            vector_store = _build_vector_store_from_env()
            if vector_store:
                try:
                    embeddings_db = AssetEmbeddings(vector_store=vector_store)
                    embeddings_db.load_from_vector_store()
                    print("[PIPELINE] Loaded asset embeddings from vector store")
                except Exception as exc:
                    embeddings_db = None
                    warn_msg = f"Failed to load asset embeddings from vector store: {exc}"
                    print(f"[PIPELINE] Warning: {warn_msg}")
                    result["warnings"].append(warn_msg)

            if embeddings_db is None:
                warn_msg = (
                    "Asset embeddings not available; matching will rely on catalog "
                    "tags and rules."
                )
                print(f"[PIPELINE] Warning: {warn_msg}")
                result["warnings"].append(warn_msg)

            auto_select_floor = _get_float_env("ASSET_MIN_AUTO_SELECT", 0.35)
            auto_select_ceiling = _get_float_env("ASSET_MAX_AUTO_SELECT", 0.5)
            debug_matching = os.getenv("ASSET_MATCH_DEBUG", "")
            debug_matching_enabled = debug_matching.lower() in {"1", "true", "yes", "on"}

            asset_matcher = AssetMatcher(
                asset_catalog,
                embeddings_db=embeddings_db,
                auto_select_floor=auto_select_floor,
                auto_select_ceiling=auto_select_ceiling,
                debug=debug_matching_enabled,
            )

            if debug_matching_enabled:
                print("[PIPELINE] Asset match debug logging enabled")

            # Normalize object descriptions for matching
            normalized_objects = []
            for obj in scene_plan.get("object_inventory", []):
                if not (obj.get("description") and str(obj.get("description")).strip()):
                    obj = dict(obj)
                    obj["description"] = AssetMatcher.build_description(obj)
                normalized_objects.append(obj)

            # Match assets
            match_results = asset_matcher.match_batch(normalized_objects)
            matched_assets = asset_matcher.to_matched_assets(match_results)
            match_debug_records = [
                r.debug for r in match_results.values() if r.debug
            ]
            unmatched_object_ids = [
                obj_id
                for obj_id, asset in matched_assets.items()
                if not asset.get("chosen_path")
            ]
            matched_count = sum(
                1 for m in matched_assets.values() if m.get("chosen_path")
            )

            result["phases"]["matching"] = {
                "success": True,
                "total_objects": len(match_results),
                "matched": matched_count,
                "unmatched": len(unmatched_object_ids),
            }
            if match_debug_records:
                result["phases"]["matching"]["debug"] = match_debug_records
            if unmatched_object_ids:
                result["phases"]["matching"]["unmatched_ids"] = unmatched_object_ids

            # Collect warnings
            for obj_result in match_results.values():
                if obj_result.warnings:
                    result["warnings"].extend(obj_result.warnings)

            # Save matched assets
            matched_assets_path = output_dir / "matched_assets.json"
            with open(matched_assets_path, "w") as f:
                json.dump(matched_assets, f, indent=2)

            if match_debug_records:
                debug_path = output_dir / "asset_match_debug.json"
                with open(debug_path, "w") as f:
                    json.dump(match_debug_records, f, indent=2)

            print(f"[PIPELINE] Matched {result['phases']['matching']['matched']}/{len(match_results)} assets")

            # ============================================================
            # PHASE 3: Recipe Compilation
            # ============================================================
            update_firestore_status(scene_id, "processing", "compiling")
            print("[PIPELINE] Phase 3: Recipe Compilation")

            compile_output_dir = output_dir / "compiled"
            compile_output_dir.mkdir(parents=True, exist_ok=True)

            # Get asset root from environment or use default
            asset_root = os.getenv("ASSET_ROOT", "/mnt/assets")
            pack_name = asset_catalog.pack_name

            # Normalize asset root
            if Path(asset_root).name == pack_name:
                asset_root = str(Path(asset_root).parent)

            compiler = RecipeCompiler(
                CompilerConfig(
                    output_dir=str(compile_output_dir),
                    asset_root=asset_root,
                )
            )

            compilation_result = compiler.compile(
                scene_plan,
                matched_assets,
                metadata={
                    "recipe_id": scene_id,
                    "source_image_uri": image_uri,
                    "description": task_intent,
                }
            )

            if compilation_result.warnings:
                result["warnings"].extend(compilation_result.warnings)
            if compilation_result.errors:
                result["errors"].extend(compilation_result.errors)

            if not compilation_result.success:
                raise RuntimeError("Recipe compilation failed")

            # Load the compiled recipe
            with open(compilation_result.recipe_path) as f:
                recipe = json.load(f)

            result["phases"]["compilation"] = {
                "success": True,
                "recipe_path": compilation_result.recipe_path,
                "scene_path": compilation_result.scene_path,
                "objects_in_recipe": len(recipe.get("objects", [])),
            }

            print(f"[PIPELINE] Compiled recipe with {len(recipe.get('objects', []))} objects")

            # ============================================================
            # PHASE 4: Validation
            # ============================================================
            update_firestore_status(scene_id, "processing", "validating")
            print("[PIPELINE] Phase 4: Validation")

            # Import validation functions
            from api.validation import (
                validate_usd_structure,
                validate_asset_references,
                validate_physics_config,
                validate_semantics,
            )

            usd_result = validate_usd_structure(compilation_result.scene_path)
            ref_result = validate_asset_references(recipe, asset_root)
            physics_result = validate_physics_config(recipe)
            semantics_result = validate_semantics(recipe)

            validation_errors = []
            validation_warnings = []

            validation_errors.extend(usd_result.get("errors", []))
            validation_warnings.extend(usd_result.get("warnings", []))
            validation_warnings.extend(physics_result.get("warnings", []))

            if ref_result.get("missing"):
                for missing in ref_result["missing"]:
                    validation_errors.append(
                        "Missing asset: "
                        f"{missing.get('asset_path') or 'not selected'} "
                        f"({missing.get('object_id', 'unknown')})"
                    )

            if unmatched_object_ids:
                for obj_id in unmatched_object_ids:
                    validation_errors.append(
                        f"Unmatched object: {obj_id} has no chosen asset"
                    )

            if semantics_result.get("missing_semantics"):
                for obj_id in semantics_result["missing_semantics"]:
                    validation_warnings.append(f"Missing semantics: {obj_id}")

            is_valid = len(validation_errors) == 0

            result["phases"]["validation"] = {
                "success": is_valid,
                "usd_structure": usd_result,
                "asset_references": ref_result,
                "physics": physics_result,
                "semantics": semantics_result,
                "errors": validation_errors,
                "warnings": validation_warnings,
            }

            result["warnings"].extend(validation_warnings)
            if validation_errors:
                result["errors"].extend(validation_errors)

            # Save validation report
            validation_report = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "scene_id": scene_id,
                "valid": is_valid,
                "ready_for_simulation": is_valid and usd_result.get("valid", False),
                "usd_structure": usd_result,
                "asset_references": ref_result,
                "physics": physics_result,
                "semantics": semantics_result,
                "errors": validation_errors,
                "warnings": validation_warnings,
            }

            validation_path = output_dir / "validation_report.json"
            with open(validation_path, "w") as f:
                json.dump(validation_report, f, indent=2)

            print(f"[PIPELINE] Validation: valid={is_valid}, errors={len(validation_errors)}, warnings={len(validation_warnings)}")

            # ============================================================
            # PHASE 5: Upload Results
            # ============================================================
            print("[PIPELINE] Phase 5: Uploading Results")

            gcs_prefix = f"gs://{bucket}/{output_prefix}"

            # Upload all output files
            uploaded_files = upload_directory_to_gcs(str(output_dir), gcs_prefix)

            # Also upload the compiled directory
            if compile_output_dir.exists():
                compiled_files = upload_directory_to_gcs(str(compile_output_dir), f"{gcs_prefix}/compiled")
                uploaded_files.extend(compiled_files)

            result["artifacts"] = {
                "scene_plan": f"{gcs_prefix}/scene_plan.json",
                "matched_assets": f"{gcs_prefix}/matched_assets.json",
                "recipe": f"{gcs_prefix}/compiled/recipe.json",
                "scene_usd": f"{gcs_prefix}/compiled/scene.usda",
                "validation_report": f"{gcs_prefix}/validation_report.json",
                "all_files": uploaded_files,
            }
            if match_debug_records:
                result["artifacts"]["asset_match_debug"] = f"{gcs_prefix}/asset_match_debug.json"

            # Upload final result summary
            result["success"] = True
            result["completed_at"] = datetime.utcnow().isoformat() + "Z"

            result_json = json.dumps(result, indent=2, default=str)
            upload_string_to_gcs(result_json, f"{gcs_prefix}/pipeline_result.json")

            print(f"[PIPELINE] Uploaded {len(uploaded_files)} files to {gcs_prefix}")

            # Update Firestore with success
            update_firestore_status(
                scene_id,
                "complete",
                "done",
                artifacts=result["artifacts"]
            )

            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            result["success"] = False
            result["completed_at"] = datetime.utcnow().isoformat() + "Z"

            print(f"[PIPELINE] Error: {error_msg}")
            traceback.print_exc()

            # Try to upload error result
            try:
                gcs_prefix = f"gs://{bucket}/{output_prefix}"
                result_json = json.dumps(result, indent=2, default=str)
                upload_string_to_gcs(result_json, f"{gcs_prefix}/pipeline_result.json")
            except Exception as upload_error:
                print(f"[PIPELINE] Failed to upload error result: {upload_error}")

            # Update Firestore with failure
            update_firestore_status(scene_id, "failed", error=error_msg)

            return result


def main():
    parser = argparse.ArgumentParser(description="BlueprintRecipe Pipeline Job")
    parser.add_argument("--image-uri", help="GCS URI of the source image")
    parser.add_argument("--scene-id", help="Scene ID (extracted from path if not provided)")
    parser.add_argument("--bucket", help="GCS bucket for output")
    parser.add_argument("--environment-type", help="Environment type hint")
    parser.add_argument("--task-intent", help="Task intent description")
    parser.add_argument("--target-policies", help="Comma-separated target policies")
    args = parser.parse_args()

    # Get values from args or environment
    image_uri = args.image_uri or os.environ.get("IMAGE_URI")
    scene_id = args.scene_id or os.environ.get("SCENE_ID")
    bucket = args.bucket or os.environ.get("BUCKET", "blueprint-8c1ca.appspot.com")
    environment_type = args.environment_type or os.environ.get("ENVIRONMENT_TYPE")
    task_intent = args.task_intent or os.environ.get("TASK_INTENT", "general scene for robot manipulation")
    target_policies_str = args.target_policies or os.environ.get("TARGET_POLICIES", "pick_place")

    if not image_uri:
        print("[PIPELINE] Error: IMAGE_URI is required")
        sys.exit(1)

    # Extract scene_id from path if not provided
    if not scene_id:
        scene_id = extract_scene_id(image_uri)

    # Parse target policies
    target_policies = [p.strip() for p in target_policies_str.split(",") if p.strip()]

    print(f"[PIPELINE] Starting BlueprintRecipe Pipeline")
    print(f"[PIPELINE] Image URI: {image_uri}")
    print(f"[PIPELINE] Scene ID: {scene_id}")
    print(f"[PIPELINE] Bucket: {bucket}")
    print(f"[PIPELINE] Environment: {environment_type or 'auto-detect'}")
    print(f"[PIPELINE] Task Intent: {task_intent}")
    print(f"[PIPELINE] Target Policies: {target_policies}")
    print(f"[PIPELINE] Timestamp: {datetime.utcnow().isoformat()}Z")

    # Run pipeline
    result = run_pipeline(
        image_uri=image_uri,
        scene_id=scene_id,
        bucket=bucket,
        environment_type=environment_type,
        task_intent=task_intent,
        target_policies=target_policies,
    )

    # Exit with appropriate code
    if result["success"]:
        print("[PIPELINE] Pipeline completed successfully")
        print(f"[PIPELINE] Artifacts: {json.dumps(result.get('artifacts', {}), indent=2)}")
        sys.exit(0)
    else:
        print(f"[PIPELINE] Pipeline failed: {result.get('errors', [])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
