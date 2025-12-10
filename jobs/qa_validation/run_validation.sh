#!/usr/bin/env bash
set -euo pipefail

echo "[QA] Starting QA Validation job"
echo "[QA] Job ID: ${JOB_ID:-unknown}"
echo "[QA] Timestamp: $(date -Iseconds)"

# Required environment variables
: "${JOB_ID:?Env JOB_ID is required}"
: "${BUCKET:?Env BUCKET is required}"
: "${SCENE_PATH:?Env SCENE_PATH is required}"

# Optional environment variables
OUTPUT_PREFIX="${OUTPUT_PREFIX:-jobs/${JOB_ID}/qa}"

echo "[QA] Configuration:"
echo "  - BUCKET: ${BUCKET}"
echo "  - SCENE_PATH: ${SCENE_PATH}"
echo "  - OUTPUT_PREFIX: ${OUTPUT_PREFIX}"

# Run validation
python /app/validate_scene.py \
    --job-id "${JOB_ID}" \
    --bucket "${BUCKET}" \
    --scene-path "${SCENE_PATH}" \
    --output-prefix "${OUTPUT_PREFIX}"

echo "[QA] Job complete"
