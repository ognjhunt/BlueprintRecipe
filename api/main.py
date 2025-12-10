"""
BlueprintRecipe API Service

FastAPI service for the BlueprintRecipe pipeline.
Handles:
- Job creation and management
- Asset search
- Recipe preview generation
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


# In-memory job store (use database in production)
jobs_store: dict[str, dict[str, Any]] = {}


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Job endpoints
@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks
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

    jobs_store[job_id] = job

    # Start background processing
    background_tasks.add_task(process_job, job_id)

    return JobResponse(**job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and details."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]
    return JobResponse(**job)


@app.post("/jobs/{job_id}/upload-image")
async def upload_image(job_id: str, file: UploadFile = File(...)):
    """Upload source image for a job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    # Save image (in production, upload to GCS)
    job = jobs_store[job_id]

    # Store image reference
    job["source_image"] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": 0  # Would be actual size
    }
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"

    return {"message": "Image uploaded", "job_id": job_id}


@app.post("/jobs/{job_id}/approve", response_model=JobResponse)
async def approve_selections(
    job_id: str,
    request: ApproveSelectionsRequest,
    background_tasks: BackgroundTasks
):
    """Approve asset selections and continue processing."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    # Update matched assets with approved selections
    if job.get("matched_assets"):
        for obj_id, asset_path in request.selections.items():
            if obj_id in job["matched_assets"]:
                job["matched_assets"][obj_id]["chosen_path"] = asset_path
                job["matched_assets"][obj_id]["approved"] = True

    job["status"] = "compiling"
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # Continue processing
    background_tasks.add_task(continue_compilation, job_id)

    return JobResponse(**job)


@app.get("/jobs/{job_id}/scene-plan")
async def get_scene_plan(job_id: str):
    """Get the generated scene plan for a job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    if not job.get("scene_plan"):
        raise HTTPException(status_code=404, detail="Scene plan not yet generated")

    return job["scene_plan"]


@app.get("/jobs/{job_id}/matched-assets")
async def get_matched_assets(job_id: str):
    """Get matched assets for a job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    if not job.get("matched_assets"):
        raise HTTPException(status_code=404, detail="Assets not yet matched")

    return job["matched_assets"]


@app.get("/jobs/{job_id}/recipe")
async def get_recipe(job_id: str):
    """Get the generated recipe for a job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    if not job.get("recipe"):
        raise HTTPException(status_code=404, detail="Recipe not yet generated")

    return job["recipe"]


# Asset search endpoints
@app.post("/assets/search", response_model=list[AssetSearchResult])
async def search_assets(request: AssetSearchRequest):
    """Search for assets by query."""
    # In production, this would search the asset index
    # For now, return mock results
    return [
        AssetSearchResult(
            asset_id="mock_001",
            asset_path="Furniture/Chair/ModernChair.usd",
            display_name="Modern Chair",
            category="furniture",
            score=0.95,
            dimensions={"width": 0.5, "depth": 0.5, "height": 0.9}
        )
    ]


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    """Get asset details by ID."""
    # In production, look up from index
    raise HTTPException(status_code=404, detail="Asset not found")


# Background task functions
async def process_job(job_id: str):
    """Process a job through the pipeline."""
    job = jobs_store.get(job_id)
    if not job:
        return

    try:
        # Phase 1: Planning
        job["status"] = "planning"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"

        # Call scene planner (would use actual implementation)
        # scene_plan = await run_scene_planning(job)
        # job["scene_plan"] = scene_plan

        # Mock scene plan for now
        job["scene_plan"] = {
            "version": "1.0.0",
            "environment_analysis": {
                "detected_type": job["request"].get("environment_type", "kitchen"),
                "confidence": 0.9
            },
            "object_inventory": [],
            "spatial_layout": {"placements": [], "relationships": []}
        }

        # Phase 2: Asset matching
        job["status"] = "matching"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"

        # Call asset matcher (would use actual implementation)
        # matched_assets = await run_asset_matching(job["scene_plan"])
        # job["matched_assets"] = matched_assets

        job["matched_assets"] = {}

        # Check if human approval needed
        needs_approval = any(
            not m.get("auto_approved", True)
            for m in job.get("matched_assets", {}).values()
        )

        if needs_approval:
            job["status"] = "awaiting_approval"
        else:
            await continue_compilation(job_id)

    except Exception as e:
        job["status"] = "failed"
        job["errors"].append(str(e))
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


async def continue_compilation(job_id: str):
    """Continue job processing after approval."""
    job = jobs_store.get(job_id)
    if not job:
        return

    try:
        # Phase 3: Recipe compilation
        job["status"] = "compiling"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"

        # Call recipe compiler (would use actual implementation)
        # recipe = await compile_recipe(job["scene_plan"], job["matched_assets"])
        # job["recipe"] = recipe

        job["recipe"] = {"version": "1.0.0", "metadata": {}, "objects": []}

        # Phase 4: Generate outputs
        if job["request"].get("generate_replicator"):
            # Generate Replicator configs
            job["artifacts"]["replicator_config"] = f"gs://bucket/jobs/{job_id}/replicator/"

        if job["request"].get("generate_isaac_lab"):
            # Generate Isaac Lab task
            job["artifacts"]["isaac_lab_task"] = f"gs://bucket/jobs/{job_id}/isaac_lab/"

        # Phase 5: Validation
        job["status"] = "validating"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"

        # Run QA checks (would use actual implementation)
        # qa_report = await run_qa_validation(job)
        # job["artifacts"]["qa_report"] = qa_report_path

        job["artifacts"]["qa_report"] = f"gs://bucket/jobs/{job_id}/qa/report.json"

        # Complete
        job["status"] = "complete"
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"

    except Exception as e:
        job["status"] = "failed"
        job["errors"].append(str(e))
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
