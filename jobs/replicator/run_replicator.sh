#!/usr/bin/env bash
set -euo pipefail

echo "[REPLICATOR] Starting Replicator job"
echo "[REPLICATOR] Job ID: ${JOB_ID:-unknown}"
echo "[REPLICATOR] Timestamp: $(date -Iseconds)"

# Required environment variables
: "${JOB_ID:?Env JOB_ID is required}"
: "${BUCKET:?Env BUCKET is required}"
: "${RECIPE_PATH:?Env RECIPE_PATH is required}"

# Optional environment variables
POLICY_ID="${POLICY_ID:-dexterous_pick_place}"
NUM_FRAMES="${NUM_FRAMES:-1000}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-jobs/${JOB_ID}/replicator}"

echo "[REPLICATOR] Configuration:"
echo "  - BUCKET: ${BUCKET}"
echo "  - RECIPE_PATH: ${RECIPE_PATH}"
echo "  - POLICY_ID: ${POLICY_ID}"
echo "  - NUM_FRAMES: ${NUM_FRAMES}"
echo "  - OUTPUT_PREFIX: ${OUTPUT_PREFIX}"

# Generate Replicator bundle
python /app/generate_replicator_bundle.py \
    --job-id "${JOB_ID}" \
    --bucket "${BUCKET}" \
    --recipe-path "${RECIPE_PATH}" \
    --policy-id "${POLICY_ID}" \
    --num-frames "${NUM_FRAMES}" \
    --output-prefix "${OUTPUT_PREFIX}"

echo "[REPLICATOR] Job complete"
