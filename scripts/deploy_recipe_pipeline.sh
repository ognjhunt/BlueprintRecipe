#!/usr/bin/env bash
# Deploy BlueprintRecipe Pipeline to Google Cloud
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - Docker installed (for local builds) or Cloud Build enabled
#   - Required APIs enabled (see below)
#
# Usage:
#   ./scripts/deploy_recipe_pipeline.sh

set -euo pipefail

# Configuration
PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-blueprint-8c1ca.appspot.com}"
REPO_NAME="${REPO_NAME:-blueprint}"
JOB_NAME="${JOB_NAME:-recipe-pipeline-job}"
WORKFLOW_NAME="${WORKFLOW_NAME:-recipe-pipeline}"
TRIGGER_NAME="${TRIGGER_NAME:-recipe-pipeline-trigger}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-744608654760-compute@developer.gserviceaccount.com}"

# Image URL
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${JOB_NAME}:latest"

echo "=========================================="
echo "BlueprintRecipe Pipeline Deployment"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job: ${JOB_NAME}"
echo "Image: ${IMAGE_URL}"
echo "=========================================="

# Step 0: Enable required APIs
echo ""
echo "[Step 0] Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    workflows.googleapis.com \
    eventarc.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

# Step 1: Create Artifact Registry repository if needed
echo ""
echo "[Step 1] Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe "${REPO_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || \
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"

# Step 2: Build and push Docker image
echo ""
echo "[Step 2] Building and pushing Docker image..."
cd "$(dirname "$0")/.."

# Use Cloud Build for faster builds (no local Docker needed)
gcloud builds submit \
    --tag="${IMAGE_URL}" \
    --project="${PROJECT_ID}" \
    --timeout=20m \
    -f jobs/recipe_pipeline/Dockerfile \
    .

# Step 3: Create or update Cloud Run Job
echo ""
echo "[Step 3] Creating/updating Cloud Run Job..."

# Check if job exists
if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" --project="${PROJECT_ID}" 2>/dev/null; then
    echo "Updating existing job..."
    gcloud run jobs update "${JOB_NAME}" \
        --image="${IMAGE_URL}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --memory=8Gi \
        --cpu=2 \
        --task-timeout=60m \
        --max-retries=1 \
        --set-env-vars="BUCKET=${BUCKET}" \
        --set-env-vars="FIRESTORE_PROJECT_ID=${PROJECT_ID}" \
        --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest"
else
    echo "Creating new job..."
    gcloud run jobs create "${JOB_NAME}" \
        --image="${IMAGE_URL}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --memory=8Gi \
        --cpu=2 \
        --task-timeout=60m \
        --max-retries=1 \
        --set-env-vars="BUCKET=${BUCKET}" \
        --set-env-vars="FIRESTORE_PROJECT_ID=${PROJECT_ID}" \
        --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest"
fi

# Step 4: Deploy Workflow
echo ""
echo "[Step 4] Deploying workflow..."
gcloud workflows deploy "${WORKFLOW_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --source="workflows/recipe-pipeline-async.yaml" \
    --service-account="${SERVICE_ACCOUNT}"

# Step 5: Create Eventarc trigger
echo ""
echo "[Step 5] Creating Eventarc trigger..."

# Delete existing trigger if it exists (to update configuration)
gcloud eventarc triggers delete "${TRIGGER_NAME}" \
    --location="us" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || true

# Create new trigger
gcloud eventarc triggers create "${TRIGGER_NAME}" \
    --location="us" \
    --project="${PROJECT_ID}" \
    --destination-workflow="${WORKFLOW_NAME}" \
    --destination-workflow-location="${REGION}" \
    --event-filters="type=google.cloud.storage.object.v1.finalized" \
    --event-filters="bucket=${BUCKET}" \
    --service-account="${SERVICE_ACCOUNT}"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "The pipeline will auto-trigger when you upload an image to:"
echo "  gs://${BUCKET}/scenes/{scene_id}/images/{filename}.jpg"
echo ""
echo "Output will be written to:"
echo "  gs://${BUCKET}/scenes/{scene_id}/recipe/"
echo ""
echo "To test manually:"
echo "  gsutil cp test_image.jpg gs://${BUCKET}/scenes/test-scene-123/images/test.jpg"
echo ""
echo "To check job status:"
echo "  gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo ""
echo "To view logs:"
echo "  gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}' --limit=50"
echo ""
