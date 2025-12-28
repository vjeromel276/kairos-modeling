# Dockerfile for Kairos daily_download.py
# Downloads SHARADAR SEP data for a single date and saves as Parquet

# Use official Python slim image - much smaller than full Python or conda
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies first (cached layer - won't rebuild unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the script we need
COPY daily_download.py .

# Create the output directory structure
RUN mkdir -p /app/scripts/sep_dataset/daily_downloads

# Environment variable for API key (will be passed at runtime)
# Do NOT put actual key here - it goes in .env or docker run command
ENV NASDAQ_DATA_LINK_API_KEY=""

# Default command shows help if no date provided
ENTRYPOINT ["python", "daily_download.py"]
CMD ["--help"]
