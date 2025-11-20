#!/bin/bash
# Automated daily data update + feature regeneration
# Run this script daily (via cron) to keep data current

set -e  # Exit on any error

LOG_FILE="logs/daily_update_$(date +%Y%m%d).log"
mkdir -p logs

echo "========================================" | tee -a $LOG_FILE
echo "Daily Update Started: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 1. Download yesterday's data
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
echo "📥 Downloading data for $YESTERDAY..." | tee -a $LOG_FILE
python scripts/daily_download.py --date $YESTERDAY 2>&1 | tee -a $LOG_FILE

# 2. Merge into DuckDB
echo "🔄 Merging into database..." | tee -a $LOG_FILE
python scripts/merge_daily_download_duck.py \
    --update-golden ./data/kairos.duckdb \
    --daily-dir ./scripts/sep_dataset/daily_downloads/ 2>&1 | tee -a $LOG_FILE

# 3. Regenerate features
echo "⚙️  Regenerating features..." | tee -a $LOG_FILE
python features/price_action_features.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE
python features/trend_features.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE
python features/statistical_features.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE
python features/volume_volatility_features.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE
python features/price_shape_features.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE
python features/generate_targets.py --db ./data/kairos.duckdb 2>&1 | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "✅ Daily Update Complete: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
