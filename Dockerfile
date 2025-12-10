# BlueprintRecipe Pipeline Job
# Processes images through scene planning, asset matching, compilation, and validation
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    pyyaml>=6.0 \
    google-cloud-storage>=2.13.0 \
    google-cloud-firestore>=2.13.0 \
    google-generativeai>=0.3.0 \
    sentence-transformers>=2.2.0

# 1. Copy the entire project context first
COPY . /app/

# 2. [FIX] Explicitly copy the script from the subfolder to the app root
# This ensures it exists at /app/run_pipeline.sh
COPY jobs/recipe_pipeline/run_pipeline.sh /app/run_pipeline.sh

# Make shell script executable
RUN chmod +x /app/run_pipeline.sh

# Set Python path to include root and src
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# Set default environment variables
ENV BUCKET="blueprint-8c1ca.appspot.com"
ENV TASK_INTENT="general scene for robot manipulation"
ENV TARGET_POLICIES="pick_place"

ENTRYPOINT ["/app/run_pipeline.sh"]