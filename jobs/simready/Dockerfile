# SimReady Job - Asset preparation and scene assembly
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
    google-generativeai \
    google-cloud-storage \
    pyyaml

# Copy source
COPY prepare_simready.py /app/prepare_simready.py
COPY run_simready.sh /app/run_simready.sh

RUN chmod +x /app/run_simready.sh

ENTRYPOINT ["/app/run_simready.sh"]
