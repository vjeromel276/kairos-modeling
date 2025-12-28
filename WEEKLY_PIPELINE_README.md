# Weekly Pipeline Setup

## Overview

`weekly_pipeline.sh` automates your Friday data refresh:
1. Downloads Monday-Friday market data for the current week
2. Merges all data into DuckDB

## Files

```
kairos_phase4/
├── weekly_pipeline.sh      # Main automation script
├── docker/
│   ├── download/           # Download container
│   │   ├── .env            # API key (already configured)
│   │   └── ...
│   └── merge/              # Merge container
│       └── ...
└── logs/                   # Pipeline logs (created automatically)
```

## Manual Test

Before setting up cron, test the script manually:

```bash
cd /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme\ Pro/kairos_phase4

# Make executable
chmod +x weekly_pipeline.sh

# Run it (will download this week's data)
./weekly_pipeline.sh
```

Check the log file in `logs/` to verify it worked.

## Cron Setup (Friday 9pm)

### Step 1: Open crontab editor

```bash
crontab -e
```

### Step 2: Add this line

```cron
0 21 * * 5 /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme\ Pro/kairos_phase4/weekly_pipeline.sh
```

**Cron format breakdown:**
- `0` — minute 0
- `21` — hour 21 (9pm)
- `*` — any day of month
- `*` — any month
- `5` — Friday (0=Sunday, 5=Friday)

### Step 3: Verify cron is set

```bash
crontab -l
```

## Troubleshooting

### Script doesn't run from cron

Cron has a limited PATH. If docker isn't found, add full paths:

```bash
# Find docker path
which docker
# Usually: /usr/bin/docker
```

Then edit the script to use `/usr/bin/docker` instead of just `docker`.

### Permission denied

Make sure the script is executable:
```bash
chmod +x weekly_pipeline.sh
```

### Check cron logs

```bash
grep CRON /var/log/syslog | tail -20
```

### Docker permission denied from cron

Cron may not have access to Docker. Fix by ensuring your user is in the docker group:

```bash
groups  # Should show 'docker'
```

If not:
```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

## Log Files

Logs are stored in `logs/weekly_pipeline_YYYYMMDD_HHMMSS.log`

View recent logs:
```bash
ls -lt logs/ | head -5
cat logs/weekly_pipeline_*.log | tail -50
```

## Customization

### Change schedule

Edit crontab. Examples:
- Saturday 8am: `0 8 * * 6`
- Sunday 6pm: `0 18 * * 0`
- Every day at midnight: `0 0 * * *`

### Download different date range

Edit `weekly_pipeline.sh` and modify the date calculation logic.

### Add email notifications

Add to crontab line:
```cron
MAILTO="your@email.com"
0 21 * * 5 /path/to/weekly_pipeline.sh
```
