#!/bin/bash

# Get the full path to the project directory
PROJECT_DIR="$(pwd)"
CONDA_ENV="kairos-gpu"

# Create cron entry
CRON_ENTRY="0 6 * * * cd $PROJECT_DIR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && ./daily_update_pipeline.sh"

# Add to crontab (if not already there)
(crontab -l 2>/dev/null | grep -v "daily_update_pipeline.sh"; echo "$CRON_ENTRY") | crontab -

echo "✅ Cron job configured!"
echo ""
echo "📅 Schedule: Daily at 6:00 AM"
echo "📂 Working directory: $PROJECT_DIR"
echo "🐍 Conda environment: $CONDA_ENV"
echo ""
echo "To view your cron jobs: crontab -l"
echo "To edit cron jobs: crontab -e"
echo "To remove this job: crontab -e (then delete the line)"
