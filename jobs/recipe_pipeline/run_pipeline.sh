#!/usr/bin/env bash
set -euo pipefail

echo "[RECIPE-PIPELINE] Starting BlueprintRecipe Pipeline Job"
echo "[RECIPE-PIPELINE] Timestamp: $(date -Iseconds)"

# Required environment variables
: "${IMAGE_URI:?Env IMAGE_URI is required}"

# Optional environment variables with defaults
BUCKET="${BUCKET:-blueprint-8c1ca.appspot.com}"
SCENE_ID="${SCENE_ID:-}"
ENVIRONMENT_TYPE="${ENVIRONMENT_TYPE:-}"
TASK_INTENT="${TASK_INTENT:-general scene for robot manipulation}"
TARGET_POLICIES="${TARGET_POLICIES:-pick_place}"

echo "[RECIPE-PIPELINE] Configuration:"
echo "  - IMAGE_URI: ${IMAGE_URI}"
echo "  - BUCKET: ${BUCKET}"
echo "  - SCENE_ID: ${SCENE_ID:-<auto-extract from path>}"
echo "  - ENVIRONMENT_TYPE: ${ENVIRONMENT_TYPE:-<auto-detect>}"
echo "  - TASK_INTENT: ${TASK_INTENT}"
echo "  - TARGET_POLICIES: ${TARGET_POLICIES}"

# Sanity check to ensure the Python entrypoint exists in the container
if [[ ! -f "/app/run_pipeline.py" ]]; then
  echo "[RECIPE-PIPELINE] ERROR: /app/run_pipeline.py not found. Current /app contents:" >&2
  ls -la /app >&2
  exit 2
fi

# Run the pipeline
python /app/run_pipeline.py \
    --image-uri "${IMAGE_URI}" \
    --bucket "${BUCKET}" \
    ${SCENE_ID:+--scene-id "${SCENE_ID}"} \
    ${ENVIRONMENT_TYPE:+--environment-type "${ENVIRONMENT_TYPE}"} \
    --task-intent "${TASK_INTENT}" \
    --target-policies "${TARGET_POLICIES}"

echo "[RECIPE-PIPELINE] Job complete"
