# BlueprintRecipe Pipeline Job
# Processes images through scene planning, asset matching, compilation, and validation
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement spec and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    sentence-transformers>=2.2.0

# Copy the entire project context
COPY . /app/

# Ensure pipeline entrypoint assets are in predictable locations
COPY jobs/recipe_pipeline/run_pipeline.sh /app/run_pipeline.sh
COPY jobs/recipe_pipeline/run_pipeline.py /app/run_pipeline.py
RUN chmod +x /app/run_pipeline.sh && \
    echo "=== Contents of /app/ ===" && \
    ls -la /app/run_pipeline.* && \
    echo "=== Contents of /app/jobs/recipe_pipeline/ ===" && \
    ls -la /app/jobs/recipe_pipeline/

# Set Python path to include root and src
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# Set default environment variables
ENV BUCKET="blueprint-8c1ca.appspot.com"
ENV TASK_INTENT="general scene for robot manipulation"
ENV TARGET_POLICIES="pick_place"

ENTRYPOINT ["/app/run_pipeline.sh"]