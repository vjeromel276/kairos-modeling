# Dockerfile for Kairos merge_daily_download_duck.py
# Merges downloaded Parquet files into DuckDB database

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the merge script
COPY merge_daily_download_duck.py .

# Default command shows help
ENTRYPOINT ["python", "merge_daily_download_duck.py"]
CMD ["--help"]
