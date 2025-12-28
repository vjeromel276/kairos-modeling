# Docker Setup for merge_daily_download_duck.py

## Overview

This containerizes the `merge_daily_download_duck.py` script, which:
1. Reads Parquet files from the daily downloads directory
2. Merges them into the `sep_base` table in DuckDB
3. Updates `sep_base_common` (incremental or full rebuild)
4. Deletes the processed Parquet files

## Files

```
docker/merge/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

### Step 1: Copy the script into the build context

From your kairos_phase4 project root:

```bash
cp scripts/merge_daily_download_duck.py docker/merge/
```

### Step 2: Build the Docker image

```bash
cd docker/merge
docker build -t kairos-merge:v1 .
```

### Step 3: Verify the build

```bash
docker run kairos-merge:v1
```

This should display the help message.

### Step 4: Run the merge

```bash
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/kairos_phase4/data:/data \
  -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/downloads \
  kairos-merge:v1 \
  --update-golden /data/kairos.duckdb \
  --daily-dir /downloads
```

**Explanation of flags:**
- `--user $(id -u):$(id -g)` — run as your user (avoids root-owned files)
- `-v .../data:/data` — mounts your data directory containing kairos.duckdb
- `-v .../daily_downloads:/downloads` — mounts the Parquet files directory
- `--update-golden /data/kairos.duckdb` — path to DuckDB (inside container)
- `--daily-dir /downloads` — path to Parquet files (inside container)

### Optional: Full rebuild of sep_base_common

Add the `--rebuild-common` flag:

```bash
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/kairos_phase4/data:/data \
  -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/downloads \
  kairos-merge:v1 \
  --update-golden /data/kairos.duckdb \
  --daily-dir /downloads \
  --rebuild-common
```

## Important Notes

### Database Locking

DuckDB uses file locking. Ensure no other process (Python, DuckDB CLI, etc.) 
has the database open when running the merge container.

### File Deletion

The merge script **deletes** Parquet files after successfully merging them.
This is intentional (prevents re-processing). If you want to keep the files,
modify the script or make a backup first.

### Database Size

Your DuckDB file is ~107GB. This is handled via volume mount — it is NOT 
copied into the container. The container reads/writes directly to your 
host filesystem.

## Verification

After running, verify the data was merged:

```bash
python3 -c "
import duckdb
con = duckdb.connect('data/kairos.duckdb', read_only=True)
print('Last sep_base date:', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0])
con.close()
"
```
