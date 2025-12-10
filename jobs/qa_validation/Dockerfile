# QA Validation Job - Scene validation with Isaac Sim
# Note: This requires Isaac Sim runtime for full validation
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    google-cloud-storage \
    google-genai

# Copy source
COPY validate_scene.py /app/validate_scene.py
COPY gemini_qa.py /app/gemini_qa.py
COPY run_validation.sh /app/run_validation.sh

RUN chmod +x /app/run_validation.sh

ENTRYPOINT ["/app/run_validation.sh"]
