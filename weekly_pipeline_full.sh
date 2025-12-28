#!/bin/bash
# weekly_pipeline_full.sh
# Complete weekly data refresh for Kairos
#
# Runs every Friday at 9pm to:
# 1. Download Monday-Friday market data
# 2. Merge into DuckDB
# 3. Rebuild universe and academic base
# 4. Generate all features, factors, composites
# 5. Assemble feature matrix
# 6. Cleanup

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Update these paths for your system
# =============================================================================
PROJECT_ROOT="/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4"
DOWNLOADS_DIR="${PROJECT_ROOT}/scripts/sep_dataset/daily_downloads"
DATA_DIR="${PROJECT_ROOT}/data"
FEATURES_DIR="${PROJECT_ROOT}/scripts/sep_dataset/feature_sets"
ENV_FILE="${PROJECT_ROOT}/docker/download/.env"
LOG_DIR="${PROJECT_ROOT}/logs"

# Docker images
DOWNLOAD_IMAGE="kairos-download:v1"
MERGE_IMAGE="kairos-merge:v1"
PIPELINE_IMAGE="kairos-pipeline:v1"

# =============================================================================
# SETUP
# =============================================================================
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/weekly_full_pipeline_${TIMESTAMP}.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log "=============================================="
log "KAIROS WEEKLY FULL PIPELINE"
log "=============================================="
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"

# Calculate the date for feature matrix
MATRIX_DATE=$(date +%Y-%m-%d)
log "Matrix date: $MATRIX_DATE"

# =============================================================================
# PHASE A: DOWNLOAD WEEK'S DATA
# =============================================================================
log ""
log "=============================================="
log "PHASE A: DOWNLOADING DATA"
log "=============================================="

# Get current day of week (1=Monday, 7=Sunday)
DOW=$(date +%u)

# Calculate Monday of this week
if [ "$DOW" -eq 7 ]; then
    MONDAY=$(date -d "6 days ago" +%Y-%m-%d)
else
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
# PHASE B: MERGE INTO DUCKDB
# =============================================================================
log ""
log "=============================================="
log "PHASE B: MERGING INTO DUCKDB"
log "=============================================="

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
# PHASE C: CLEANUP DOWNLOADS
# =============================================================================
log ""
log "=============================================="
log "PHASE C: CLEANUP DOWNLOADS"
log "=============================================="

REMAINING_FILES=$(ls -1 "$DOWNLOADS_DIR"/*.parquet 2>/dev/null | wc -l)

if [ "$REMAINING_FILES" -gt 0 ]; then
    log "Removing $REMAINING_FILES leftover parquet files..."
    rm -f "$DOWNLOADS_DIR"/*.parquet
    log "✓ Cleanup complete"
else
    log "No files to clean up"
fi

# =============================================================================
# PHASE D: RUN FULL PIPELINE (Phases 1-5)
# =============================================================================
log ""
log "=============================================="
log "PHASE D: RUNNING PIPELINE (Phases 1-5)"
log "=============================================="

log "Starting pipeline container..."

if docker run \
    --user "$(id -u):$(id -g)" \
    -v "$DATA_DIR":/data \
    -v "$FEATURES_DIR":/features \
    "$PIPELINE_IMAGE" \
    run_pipeline.py \
    --db /data/kairos.duckdb \
    --universe /features/option_b_universe.csv \
    --date "$MATRIX_DATE" 2>&1 | tee -a "$LOG_FILE"; then
    
    log "✓ Pipeline completed successfully"
else
    log "✗ Pipeline failed!"
    exit 1
fi

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "=============================================="
log "WEEKLY PIPELINE COMPLETE"
log "=============================================="
log "Log file: $LOG_FILE"
log ""
log "To verify results:"
log "  python3 -c \"import duckdb; con=duckdb.connect('$DATA_DIR/kairos.duckdb',read_only=True); print('Latest sep_base:', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0]); print('Latest feat_matrix:', con.execute('SELECT MAX(date) FROM feat_matrix').fetchone()[0])\""
log "=============================================="
