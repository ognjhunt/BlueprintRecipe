#!/usr/bin/env bash
set -euo pipefail

echo "[SIMREADY] Starting SimReady job"
echo "[SIMREADY] Job ID: ${JOB_ID:-unknown}"
echo "[SIMREADY] Timestamp: $(date -Iseconds)"

# Required environment variables
: "${JOB_ID:?Env JOB_ID is required}"
: "${BUCKET:?Env BUCKET is required}"
: "${RECIPE_PATH:?Env RECIPE_PATH is required}"

# Optional environment variables
ASSETS_ROOT="${ASSETS_ROOT:-/mnt/gcs/assets}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-jobs/${JOB_ID}/simready}"

echo "[SIMREADY] Configuration:"
echo "  - BUCKET: ${BUCKET}"
echo "  - RECIPE_PATH: ${RECIPE_PATH}"
echo "  - ASSETS_ROOT: ${ASSETS_ROOT}"
echo "  - OUTPUT_PREFIX: ${OUTPUT_PREFIX}"

# Run the preparation script
python /app/prepare_simready.py \
    --job-id "${JOB_ID}" \
    --bucket "${BUCKET}" \
    --recipe-path "${RECIPE_PATH}" \
    --assets-root "${ASSETS_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}"

echo "[SIMREADY] Job complete"
