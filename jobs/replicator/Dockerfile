# Replicator Job - Synthetic Data Generation
# Note: This requires Isaac Sim runtime for actual SDG
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    pyyaml \
    google-cloud-storage

# Copy source
COPY generate_replicator_bundle.py /app/generate_replicator_bundle.py
COPY run_replicator.sh /app/run_replicator.sh

RUN chmod +x /app/run_replicator.sh

ENTRYPOINT ["/app/run_replicator.sh"]
