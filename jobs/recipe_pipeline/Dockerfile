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

# Copy entire project (we need src/, api/, asset_index.json, data/)
COPY . /app/

# Make shell script executable
RUN chmod +x /app/jobs/recipe_pipeline/run_pipeline.sh

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# Set default environment variables
ENV BUCKET="blueprint-8c1ca.appspot.com"
ENV TASK_INTENT="general scene for robot manipulation"
ENV TARGET_POLICIES="pick_place"

ENTRYPOINT ["/app/jobs/recipe_pipeline/run_pipeline.sh"]
