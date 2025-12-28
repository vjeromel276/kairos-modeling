# Docker Setup for daily_download.py

## Overview

This containerizes the `daily_download.py` script, which downloads SHARADAR SEP 
data for a single date from the Nasdaq Data Link API and saves it as a Parquet file.

## Files

```
docker/download/
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies (requests, pandas, pyarrow)
└── README.md         # This file
```

## Prerequisites

- Docker installed and running
- Nasdaq Data Link API key

## Setup Instructions

### Step 1: Copy the script into the docker build context

From your kairos_phase4 project root:

```bash
cp scripts/daily_download.py docker/download/
```

### Step 2: Build the Docker image

```bash
cd docker/download
docker build -t kairos-download:v1 .
```

### Step 3: Verify the build

```bash
docker run kairos-download:v1
```

This should display the help message (since no date was provided).

### Step 4: Run with actual data

```bash
docker run \
  -e NASDAQ_DATA_LINK_API_KEY="your-api-key-here" \
  -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/app/scripts/sep_dataset/daily_downloads \
  kairos-download:v1 \
  --date 2025-12-20
```

**Explanation of flags:**
- `-e NASDAQ_DATA_LINK_API_KEY=...` passes your API key into the container
- `-v /host/path:/container/path` mounts a volume so output files persist on your host
- `--date 2025-12-20` is passed to the Python script

## Using an Environment File (Recommended)

Instead of passing the API key on the command line, create a `.env` file:

```bash
# docker/download/.env
NASDAQ_DATA_LINK_API_KEY=your-actual-api-key
```

Then run with:

```bash
docker run \
  --env-file .env \
  -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/app/scripts/sep_dataset/daily_downloads \
  kairos-download:v1 \
  --date 2025-12-20
```

**Important:** Add `.env` to your `.gitignore` to avoid committing your API key.

## Verification

After running, check that the Parquet file was created:

```bash
ls -la scripts/sep_dataset/daily_downloads/
```

You should see: `SHARADAR_SEP_2025-12-20.parquet`

## Troubleshooting

**"Environment variable not found" error:**
- Make sure you passed `-e NASDAQ_DATA_LINK_API_KEY=...` or `--env-file .env`

**"Permission denied" on output directory:**
- Ensure the host directory exists and is writable
- You may need to create it first: `mkdir -p scripts/sep_dataset/daily_downloads`

**"No such file or directory" for the volume mount:**
- Use absolute paths for the volume mount, not relative paths
- Example: `-v /media/vjl2dev/.../kairos_phase4/scripts/sep_dataset/daily_downloads:/app/scripts/sep_dataset/daily_downloads`
