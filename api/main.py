"""
BlueprintRecipe API Service

FastAPI service for the BlueprintRecipe pipeline.
Handles:
- Job creation and management
- Asset search
- Recipe preview generation
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.database import (
    FirestoreJobRepository,
    InMemoryJobRepository,
    JobRepository,
)
from api.exceptions import (
    BlueprintAPIError,
    CompilationError,
    MatchingError,
    PlanningError,
    StorageError,
    ValidationError,
)
from api.validation import validate_scene
from api.storage import (
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    StorageUploadError,
    upload_artifacts,
    upload_file_to_gcs,
)
from src.asset_catalog import AssetCatalogBuilder, AssetMatcher
from src.planning import ScenePlanner
from src.recipe_compiler import RecipeCompiler
from src.recipe_compiler.compiler import CompilerConfig

# Initialize FastAPI app
app = FastAPI(
    title="BlueprintRecipe API",
    description="Scene recipe generation for Isaac Sim/Lab training",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Configure application dependencies."""
    global job_repository
    job_repository = _init_repository()
    try:
        _get_asset_matcher()
    except Exception:  # pragma: no cover - defensive load
        logging.exception("Failed to preload asset matcher; will retry on demand")


# Pydantic models
class JobCreateRequest(BaseModel):
    """Request to create a new recipe generation job."""
    source_image_uri: Optional[str] = None
    environment_type: Optional[str] = None
    task_intent: Optional[str] = None
    target_policies: list[str] = Field(default_factory=list)
    asset_packs: list[str] = Field(default_factory=lambda: ["ResidentialAssetsPack"])
    generate_replicator: bool = True
    generate_isaac_lab: bool = True
    callback_url: Optional[str] = None


class JobResponse(BaseModel):
    """Response with job details."""
    job_id: str
    status: str  # pending, planning, matching, compiling, validating, complete, failed
    created_at: str
    updated_at: str
    environment_type: Optional[str] = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validation: Optional[dict[str, Any]] = None


class AssetSearchRequest(BaseModel):
    """Request for asset search."""
    query: str
    category: Optional[str] = None
    pack_name: Optional[str] = None
    top_k: int = 10


class AssetSearchResult(BaseModel):
    """Single asset search result."""
    asset_id: str
    asset_path: str
    display_name: str
    category: str
    score: float
    dimensions: Optional[dict[str, float]] = None
    thumbnail_url: Optional[str] = None


class ApproveSelectionsRequest(BaseModel):
    """Request to approve asset selections."""
    selections: dict[str, str]  # object_id -> chosen_asset_path


job_repository: JobRepository | None = None
asset_catalog = None
asset_matcher = None
asset_lookup: dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _get_auto_select_thresholds() -> tuple[float, float]:
    """Return configured auto-select thresholds (floor, ceiling)."""

    def _coerce(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except (TypeError, ValueError):
            logger.warning("Invalid value for %s; using default %.2f", name, default)
            return default

    floor = _coerce("ASSET_MIN_AUTO_SELECT", 0.35)
    ceiling = _coerce("ASSET_MAX_AUTO_SELECT", 0.5)
    return floor, max(floor, ceiling)


def _load_catalog() -> None:
    """Load the asset catalog and matcher if not already loaded."""
    global asset_catalog, asset_matcher, asset_lookup

    if asset_catalog and asset_matcher:
        return

    catalog_path = Path(__file__).resolve().parents[1] / "asset_index.json"
    asset_catalog = AssetCatalogBuilder.load(str(catalog_path))
    floor, ceiling = _get_auto_select_thresholds()
    asset_matcher = AssetMatcher(
        asset_catalog,
        auto_select_floor=floor,
        auto_select_ceiling=ceiling,
    )
    asset_lookup = {entry.asset_id: entry for entry in asset_catalog.assets}


def _get_asset_matcher() -> AssetMatcher:
    """Ensure the asset matcher is available, loading on demand."""
    _load_catalog()
    return asset_matcher


def _init_repository() -> JobRepository:
    """Initialize a job repository based on environment configuration."""

    backend = os.getenv("JOB_REPOSITORY_BACKEND", "in_memory").lower()

    if backend == "firestore":
        project_id = os.getenv("FIRESTORE_PROJECT_ID")
        collection = os.getenv("FIRESTORE_COLLECTION", "jobs")

        if not project_id:
            raise RuntimeError("FIRESTORE_PROJECT_ID must be set for Firestore backend")

        return FirestoreJobRepository(project_id=project_id, collection=collection)

    if backend != "in_memory":
        raise RuntimeError(
            "Unsupported JOB_REPOSITORY_BACKEND. Supported values: in_memory, firestore"
        )

    return InMemoryJobRepository()


async def get_job_repository() -> JobRepository:
    if job_repository is None:
        raise HTTPException(status_code=500, detail="Job repository not initialized")
    return job_repository


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Job endpoints
@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    repository: JobRepository = Depends(get_job_repository),
):
    """Create a new recipe generation job."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"

    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "request": request.model_dump(),
        "environment_type": request.environment_type,
        "artifacts": {},
        "warnings": [],
        "errors": [],
        "scene_plan": None,
        "matched_assets": None,
        "recipe": None
    }

    await repository.create_job(job)

    # Start background processing
    background_tasks.add_task(process_job, job_id, repository)

    return JobResponse(**job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, repository: JobRepository = Depends(get_job_repository)):
    """Get job status and details."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**job)


@app.post("/jobs/{job_id}/upload-image")
async def upload_image(
    job_id: str,
    file: UploadFile = File(...),
    repository: JobRepository = Depends(get_job_repository),
):
    """Upload source image for a job."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    scene_id = job.get("scene_id") or job_id
    job["scene_id"] = scene_id

    try:
        # Determine file size without loading entirely into memory
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        uri = await upload_file_to_gcs(file, scene_id=scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (StorageUploadError, RuntimeError) as exc:
        raise StorageError(str(exc)).to_http_exception() from exc
    except Exception as exc:  # pragma: no cover - defensive
        trace_id = uuid.uuid4().hex
        logger.exception("Unexpected upload failure [trace_id=%s]", trace_id)
        raise HTTPException(
            status_code=500, detail=f"Unexpected upload failure (trace_id={trace_id})"
        ) from exc

    job.setdefault("request", {})
    job["request"]["source_image_uri"] = uri
    safe_filename = Path(file.filename or "upload.bin").name
    content_type = file.content_type or "application/octet-stream"
    job["source_image"] = {
        "filename": safe_filename,
        "content_type": content_type,
        "size": size,
        "uri": uri,
    }
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"

    await repository.update_job(job_id, job)

    return {"message": "Image uploaded", "job_id": job_id, "uri": uri}


@app.post("/jobs/{job_id}/approve", response_model=JobResponse)
async def approve_selections(
    job_id: str,
    request: ApproveSelectionsRequest,
    background_tasks: BackgroundTasks,
    repository: JobRepository = Depends(get_job_repository),
):
    """Approve asset selections and continue processing."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Update matched assets with approved selections
    if job.get("matched_assets"):
        for obj_id, asset_path in request.selections.items():
            if obj_id in job["matched_assets"]:
                job["matched_assets"][obj_id]["chosen_path"] = asset_path
                job["matched_assets"][obj_id]["approved"] = True

    job["status"] = "compiling"
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"

    await repository.update_job(job_id, job)

    # Continue processing
    background_tasks.add_task(continue_compilation, job_id, repository)

    return JobResponse(**job)


@app.get("/jobs/{job_id}/scene-plan")
async def get_scene_plan(job_id: str, repository: JobRepository = Depends(get_job_repository)):
    """Get the generated scene plan for a job."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("scene_plan"):
        raise HTTPException(status_code=404, detail="Scene plan not yet generated")

    return job["scene_plan"]


@app.get("/jobs/{job_id}/matched-assets")
async def get_matched_assets(
    job_id: str, repository: JobRepository = Depends(get_job_repository)
):
    """Get matched assets for a job."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("matched_assets"):
        raise HTTPException(status_code=404, detail="Assets not yet matched")

    return job["matched_assets"]


@app.get("/jobs/{job_id}/recipe")
async def get_recipe(job_id: str, repository: JobRepository = Depends(get_job_repository)):
    """Get the generated recipe for a job."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("recipe"):
        raise HTTPException(status_code=404, detail="Recipe not yet generated")

    return job["recipe"]


@app.get("/jobs/{job_id}/validation")
async def get_validation(
    job_id: str, repository: JobRepository = Depends(get_job_repository)
):
    """Get validation results for a job."""
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("validation"):
        raise HTTPException(status_code=404, detail="Validation not yet complete")

    return job["validation"]


# Asset search endpoints
@app.post("/assets/search", response_model=list[AssetSearchResult])
async def search_assets(request: AssetSearchRequest):
    """Search for assets by query."""
    matcher = _get_asset_matcher()

    if request.pack_name and request.pack_name != asset_catalog.pack_name:
        raise HTTPException(status_code=404, detail="Asset pack not found")

    object_spec = {
        "id": "asset_search",
        "category": request.category or "",
        "description": request.query,
        "estimated_dimensions": {},
    }

    match_result = matcher.match(object_spec, top_k=request.top_k)

    results: list[AssetSearchResult] = []
    for candidate in match_result.candidates:
        entry = asset_lookup.get(candidate.asset_id)
        if not entry:
            continue

        results.append(
            AssetSearchResult(
                asset_id=candidate.asset_id,
                asset_path=candidate.asset_path,
                display_name=entry.display_name or Path(candidate.asset_path).stem,
                category=entry.category,
                score=candidate.score,
                dimensions=entry.dimensions,
                thumbnail_url=entry.thumbnail_path,
            )
        )

    return results


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    """Get asset details by ID."""
    _load_catalog()

    entry = asset_lookup.get(asset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Asset not found")

    return {
        "asset_id": entry.asset_id,
        "asset_path": entry.relative_path,
        "display_name": entry.display_name,
        "category": entry.category,
        "subcategory": entry.subcategory,
        "description": entry.description,
        "tags": entry.tags,
        "dimensions": entry.dimensions,
        "variant_sets": entry.variant_sets,
        "materials": entry.materials,
        "simready_metadata": entry.simready_metadata,
        "default_prim": entry.default_prim,
        "thumbnail_url": entry.thumbnail_path,
        "pack_name": asset_catalog.pack_name,
    }


def _resolve_asset_root(pack_name: str) -> str:
    """Resolve the asset root, normalizing to the pack's parent directory."""
    asset_root_env = Path(os.getenv("ASSET_ROOT", "/mnt/assets"))

    # If the configured root already points at the pack, return its parent so the
    # compiler can append the pack name from the catalog without duplication.
    if asset_root_env.name == pack_name:
        return str(asset_root_env.parent)

    return str(asset_root_env)


# Background task functions
async def process_job(job_id: str, repository: JobRepository):
    """Process a job through the pipeline."""
    job = await repository.get_job(job_id)
    if not job:
        return

    try:
        # Phase 1: Planning
        job["status"] = "planning"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        job = await repository.update_job(job_id, job)

        planner = ScenePlanner()

        source_image = job["request"].get("source_image_uri")
        if not source_image or not Path(source_image).exists():
            raise PlanningError("Source image is required for scene planning")

        planning_result = planner.plan_from_image(
            image_path=source_image,
            task_intent=job["request"].get("task_intent"),
            environment_hint=job["request"].get("environment_type"),
            target_policies=job["request"].get("target_policies")
        )

        if not planning_result.success or not planning_result.scene_plan:
            raise PlanningError(planning_result.error or "Scene planning failed")

        job["scene_plan"] = planning_result.scene_plan
        if planning_result.warnings:
            job["warnings"].extend(planning_result.warnings)

        job["status"] = "matching"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        job = await repository.update_job(job_id, job)

        matcher = _get_asset_matcher()

        try:
            match_results = matcher.match_batch(
                job["scene_plan"].get("object_inventory", [])
            )
        except Exception as exc:
            raise MatchingError(str(exc) or "Asset matching failed") from exc
        matched_assets = matcher.to_matched_assets(match_results)

        job["asset_pack"] = asset_catalog.pack_name
        job["asset_root"] = _resolve_asset_root(asset_catalog.pack_name)
        job["matched_assets"] = matched_assets

        for result in match_results.values():
            if result.warnings:
                job["warnings"].extend(result.warnings)

        needs_approval = any(
            not m.get("chosen_path") or m.get("needs_selection")
            for m in job.get("matched_assets", {}).values()
        )

        if needs_approval:
            job["status"] = "awaiting_approval"
            job["updated_at"] = datetime.utcnow().isoformat() + "Z"
            await repository.update_job(job_id, job)
        else:
            await repository.update_job(job_id, job)
            await continue_compilation(job_id, repository)

    except BlueprintAPIError as exc:
        job["status"] = "failed"
        job.setdefault("errors", []).append(exc.user_message)
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)
        await _send_callback(job, "failed", repository)
    except Exception as exc:
        trace_id = uuid.uuid4().hex
        logger.exception("Unexpected job processing failure [trace_id=%s]", trace_id)
        job["status"] = "failed"
        job.setdefault("errors", []).append(
            f"Unexpected error during processing (trace_id={trace_id})"
        )
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)
        await _send_callback(job, "failed", repository)


async def continue_compilation(job_id: str, repository: JobRepository):
    """Continue job processing after approval."""
    job = await repository.get_job(job_id)
    if not job:
        return

    try:
        # Phase 3: Recipe compilation
        job["status"] = "compiling"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        job = await repository.update_job(job_id, job)

        output_dir = Path("jobs") / job_id
        _load_catalog()
        pack_name = job.get("asset_pack") or asset_catalog.pack_name

        asset_root = job.get("asset_root") or _resolve_asset_root(pack_name)

        compiler = RecipeCompiler(
            CompilerConfig(
                output_dir=str(output_dir),
                asset_root=asset_root,
            )
        )

        compilation_result = compiler.compile(
            job.get("scene_plan", {}),
            job.get("matched_assets", {}),
            metadata={
                "recipe_id": job_id,
                "source_image_uri": job["request"].get("source_image_uri"),
                "description": job["request"].get("task_intent"),
            }
        )

        if compilation_result.warnings:
            job["warnings"].extend(compilation_result.warnings)

        if compilation_result.errors:
            job["errors"].extend(compilation_result.errors)

        if not compilation_result.success:
            raise CompilationError("Recipe compilation failed")

        with open(compilation_result.recipe_path) as recipe_file:
            job["recipe"] = json.load(recipe_file)

        # Phase 4: Generate outputs
        job["artifacts"]["recipe_json"] = compilation_result.recipe_path
        job["artifacts"]["scene_usd"] = compilation_result.scene_path
        job["artifacts"]["layers"] = compilation_result.layer_paths

        qa_report_path = Path(output_dir) / "qa" / "compilation_report.json"
        job["artifacts"]["qa_report"] = str(qa_report_path)

        if compilation_result.replicator_bundle:
            job["artifacts"]["replicator_bundle"] = compilation_result.replicator_bundle

        replicator_path = output_dir / "replicator"
        if replicator_path.exists():
            job["artifacts"]["replicator_assets"] = str(replicator_path)

        if compilation_result.isaac_lab_bundle:
            job["artifacts"]["isaac_lab_bundle"] = compilation_result.isaac_lab_bundle

        isaac_lab_path = output_dir / "isaac_lab"
        if isaac_lab_path.exists():
            job["artifacts"]["isaac_lab_assets"] = str(isaac_lab_path)

        bucket_name = os.getenv("GCS_BUCKET", DEFAULT_BUCKET)
        prefix = os.getenv("GCS_PREFIX", DEFAULT_PREFIX)
        uploads_completed = False

        try:
            job["artifacts"] = await upload_artifacts(
                job_id,
                job["artifacts"],
                bucket_name=bucket_name,
                prefix=prefix,
            )
            uploads_completed = True
        except (StorageUploadError, RuntimeError, FileNotFoundError, ValueError) as exc:
            raise StorageError(f"Artifact upload failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise StorageError(f"Unexpected artifact upload error: {exc}") from exc

        if uploads_completed:
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                job["warnings"].append(
                    f"Failed to clean up local artifacts for {job_id}: {exc}"
                )

        # Phase 5: Validation
        job["status"] = "validating"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)

        # Run validation on the compiled recipe
        skip_gemini = os.getenv("SKIP_GEMINI_VALIDATION", "").lower() in ("1", "true")
        validation_result = await validate_scene(
            job_id=job_id,
            recipe=job.get("recipe", {}),
            scene_path=job["artifacts"].get("scene_usd"),
            assets_root=job.get("asset_root"),
            skip_gemini=skip_gemini,
        )

        # Store validation results
        job["validation"] = {
            "valid": validation_result.valid,
            "ready_for_simulation": validation_result.ready_for_simulation,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "report": validation_result.report,
        }

        # Add validation warnings to job warnings
        if validation_result.warnings:
            job["warnings"].extend(validation_result.warnings)

        # Handle blocking validation errors
        if not validation_result.valid:
            job["errors"].extend(validation_result.errors)
            # Mark as complete with validation warnings (non-blocking by default)
            # Set VALIDATION_ERRORS_BLOCK=1 to make validation errors block completion
            if os.getenv("VALIDATION_ERRORS_BLOCK", "").lower() in ("1", "true"):
                raise ValidationError(
                    f"Scene validation failed with {len(validation_result.errors)} errors"
                )

        # Complete
        job["status"] = "complete"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)
        await _send_callback(job, "complete", repository)

    except BlueprintAPIError as exc:
        job["status"] = "failed"
        job["errors"].append(exc.user_message)
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)
        await _send_callback(job, "failed", repository)
    except Exception as exc:
        trace_id = uuid.uuid4().hex
        logger.exception(
            "Unexpected compilation continuation failure [trace_id=%s]", trace_id
        )
        job["status"] = "failed"
        job["errors"].append(
            f"Unexpected error during compilation (trace_id={trace_id})"
        )
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await repository.update_job(job_id, job)
        await _send_callback(job, "failed", repository)


async def _send_callback(job: dict[str, Any], status: str, repository: JobRepository | None = None) -> None:
    """Send job status callbacks with retry/backoff and warning logging."""

    callback_url = job.get("request", {}).get("callback_url")
    if not callback_url:
        return

    payload = {
        "job_id": job.get("job_id"),
        "status": status,
        "artifacts": job.get("artifacts", {}),
        "warnings": job.get("warnings", []),
        "errors": job.get("errors", []),
    }

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            return
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            backoff = 2 ** attempt
            await asyncio.sleep(backoff)

    warning_message = (
        f"Callback delivery to {callback_url} failed after 3 attempts: {last_error}"
    )
    logging.warning(warning_message)
    job.setdefault("warnings", []).append(warning_message)

    if repository and job.get("job_id"):
        await repository.update_job(job["job_id"], job)


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
