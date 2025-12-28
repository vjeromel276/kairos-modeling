#!/bin/bash
# weekly_pipeline.sh
# Runs every Friday at 9pm to download and merge the week's market data
#
# Usage: ./weekly_pipeline.sh
# 
# This script:
# 1. Calculates Monday-Friday dates for the current week
# 2. Downloads each day's data using the kairos-download container
# 3. Merges all data into DuckDB using the kairos-merge container
# 4. Cleans up all parquet files from downloads directory

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Update these paths for your system
# =============================================================================
PROJECT_ROOT="/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4"
DOWNLOADS_DIR="${PROJECT_ROOT}/scripts/sep_dataset/daily_downloads"
DATA_DIR="${PROJECT_ROOT}/data"
ENV_FILE="${PROJECT_ROOT}/docker/download/.env"
LOG_DIR="${PROJECT_ROOT}/logs"

# Docker images
DOWNLOAD_IMAGE="kairos-download:v1"
MERGE_IMAGE="kairos-merge:v1"

# =============================================================================
# SETUP
# =============================================================================
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/weekly_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log "=============================================="
log "WEEKLY PIPELINE STARTED"
log "=============================================="

# =============================================================================
# CALCULATE THIS WEEK'S DATES (Monday through Friday)
# =============================================================================
# Get current day of week (1=Monday, 7=Sunday)
DOW=$(date +%u)

# Calculate Monday of this week
if [ "$DOW" -eq 7 ]; then
    # If Sunday, Monday was 6 days ago
    MONDAY=$(date -d "6 days ago" +%Y-%m-%d)
else
    # Otherwise, Monday was (DOW-1) days ago
    DAYS_SINCE_MONDAY=$((DOW - 1))
    MONDAY=$(date -d "$DAYS_SINCE_MONDAY days ago" +%Y-%m-%d)
fi

log "Monday of this week: $MONDAY"

# Generate Mon-Fri dates
DATES=()
for i in 0 1 2 3 4; do
    DATE=$(date -d "$MONDAY + $i days" +%Y-%m-%d)
    DATES+=("$DATE")
done

log "Dates to download: ${DATES[*]}"

# =============================================================================
# DOWNLOAD EACH DAY
# =============================================================================
log "----------------------------------------------"
log "PHASE 1: DOWNLOADING DATA"
log "----------------------------------------------"

DOWNLOAD_COUNT=0
DOWNLOAD_ERRORS=0

for DATE in "${DATES[@]}"; do
    log "Downloading $DATE..."
    
    if docker run \
        --env-file "$ENV_FILE" \
        --user "$(id -u):$(id -g)" \
        -v "$DOWNLOADS_DIR":/app/scripts/sep_dataset/daily_downloads \
        "$DOWNLOAD_IMAGE" \
        --date "$DATE" >> "$LOG_FILE" 2>&1; then
        
        log "✓ Downloaded $DATE"
        DOWNLOAD_COUNT=$((DOWNLOAD_COUNT + 1))
    else
        log "✗ Failed to download $DATE (may be holiday/weekend)"
        DOWNLOAD_ERRORS=$((DOWNLOAD_ERRORS + 1))
    fi
    
    # Small delay to avoid rate limiting
    sleep 2
done

log "Downloads complete: $DOWNLOAD_COUNT succeeded, $DOWNLOAD_ERRORS failed"

# =============================================================================
# MERGE INTO DUCKDB
# =============================================================================
log "----------------------------------------------"
log "PHASE 2: MERGING INTO DUCKDB"
log "----------------------------------------------"

# Check if there are any files to merge
PARQUET_COUNT=$(ls -1 "$DOWNLOADS_DIR"/*.parquet 2>/dev/null | wc -l)

if [ "$PARQUET_COUNT" -eq 0 ]; then
    log "No parquet files to merge. Skipping merge step."
else
    log "Found $PARQUET_COUNT parquet files to merge"
    
    if docker run \
        --user "$(id -u):$(id -g)" \
        -v "$DATA_DIR":/data \
        -v "$DOWNLOADS_DIR":/downloads \
        "$MERGE_IMAGE" \
        --update-golden /data/kairos.duckdb \
        --daily-dir /downloads >> "$LOG_FILE" 2>&1; then
        
        log "✓ Merge completed successfully"
    else
        log "✗ Merge failed!"
        exit 1
    fi
fi

# =============================================================================
# CLEANUP - Remove all parquet files from downloads directory
# =============================================================================
log "----------------------------------------------"
log "PHASE 3: CLEANUP"
log "----------------------------------------------"

REMAINING_FILES=$(ls -1 "$DOWNLOADS_DIR"/*.parquet 2>/dev/null | wc -l)

if [ "$REMAINING_FILES" -gt 0 ]; then
    log "Removing $REMAINING_FILES leftover parquet files..."
    rm -f "$DOWNLOADS_DIR"/*.parquet
    log "✓ Cleanup complete"
else
    log "No files to clean up"
fi

# =============================================================================
# SUMMARY
# =============================================================================
log "----------------------------------------------"
log "PIPELINE COMPLETE"
log "----------------------------------------------"
log "Log file: $LOG_FILE"

log "To verify, run:"
log "  python3 -c \"import duckdb; con=duckdb.connect('data/kairos.duckdb',read_only=True); print('Last date:', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0])\""
log "=============================================="