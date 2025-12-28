# Dockerfile for Kairos Unified Pipeline
# Contains all scripts for Phases 1-5: Data → Features → Factors → Composites → Matrix

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts maintaining directory structure
COPY scripts/ /app/scripts/

# Copy the master pipeline runner
COPY run_pipeline.py /app/

# Default entrypoint allows running any script
ENTRYPOINT ["python"]
CMD ["run_pipeline.py", "--help"]
