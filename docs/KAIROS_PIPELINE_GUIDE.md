# Kairos Pipeline Complete Guide

**Version:** 1.0  
**Last Updated:** 2026-01-18  
**Source:** Consolidated from CLAUDE.md, rebalance-design.md, WEEKLY_PIPELINE_README.md, and verified against actual source code.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites and Environment Setup](#2-prerequisites-and-environment-setup)
3. [Directory Structure](#3-directory-structure)
4. [Data Sync Process](#4-data-sync-process)
5. [Pipeline Architecture - 7 Phases](#5-pipeline-architecture---7-phases)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Weekly Automation](#7-weekly-automation)
8. [Rebalancing System](#8-rebalancing-system)
9. [Trade Execution (Alpaca)](#9-trade-execution-alpaca)
10. [Docker Deployment](#10-docker-deployment)
11. [Database Tables Reference](#11-database-tables-reference)
12. [Common Commands Quick Reference](#12-common-commands-quick-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Known Issues](#14-known-issues)

---

## 1. System Overview

Kairos Phase 4 is a quantitative trading system implementing a 7-phase ML pipeline for stock selection and weekly portfolio rebalancing. It combines the "Risk4" methodology with machine learning predictions to generate stock picks, executed via the Alpaca API.

### Core Workflow

\`\`\`
Download SHARADAR Data -> Build Features -> Generate Alpha Signals -> Detect Market Regime -> Produce Weekly Rebalance -> Execute Trades
\`\`\`

### High-Level Architecture

\`\`\`
+------------------+     +------------------+     +------------------+
|   Data Sources   | --> |   DuckDB (~115GB)|     |   Output         |
|   - SHARADAR SEP |     |   - sep_base     | --> |   - picks.csv    |
|   - SHARADAR SF1 |     |   - feat_matrix  |     |   - trades.csv   |
|   - SHARADAR SF2 |     |   - regime_hist  |     |   - summary.json |
+------------------+     +------------------+     +------------------+
         |                       |                        |
         v                       v                        v
   Nasdaq Data Link        7-Phase Pipeline         Alpaca API
\`\`\`

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Environment | Conda (\`kairos-gpu\`) |
| Database | DuckDB (~115GB at \`data/kairos.duckdb\`) |
| ML | XGBoost, Optuna (hyperparameter tuning) |
| Data Processing | Pandas, Polars, NumPy |
| Data Source | SHARADAR SEP API (via Nasdaq Data Link) |
| Broker | Alpaca API (paper and live trading) |
| Containerization | Docker (planned, not yet implemented) |

---

## 2. Prerequisites and Environment Setup

### 2.1 System Requirements

- **OS:** Linux (tested on Ubuntu 22.04+)
- **RAM:** 16GB minimum (32GB recommended for large queries)
- **Disk:** 200GB+ free space for database and models
- **GPU:** CUDA 12.1 compatible (optional, for ML training)

### 2.2 Required Software

- Python 3.11
- Conda/Miniconda
- Docker (for containerized deployment)
- Git

### 2.3 API Keys Required

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Nasdaq Data Link | \`NASDAQ_DATA_LINK_API_KEY\` | SHARADAR data access |
| Alpaca | \`ALPACA_API_KEY\` | Trade execution |
| Alpaca | \`ALPACA_SECRET_KEY\` | Trade execution |
| Alpaca | \`ALPACA_BASE_URL\` | API endpoint (paper vs live) |

### 2.4 Environment Setup

**Step 1: Clone the repository**
\`\`\`bash
git clone <repository-url> kairos_phase4
cd kairos_phase4
\`\`\`

**Step 2: Create conda environment**
\`\`\`bash
conda env create -f conda_env.yaml
conda activate kairos-gpu
\`\`\`

**Step 3: Set environment variables**
\`\`\`bash
# Add to ~/.bashrc or ~/.profile
export NASDAQ_DATA_LINK_API_KEY="your-nasdaq-api-key"
export ALPACA_API_KEY="your-alpaca-key"
export ALPACA_SECRET_KEY="your-alpaca-secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
\`\`\`

**Step 4: Verify installation**
\`\`\`bash
conda activate kairos-gpu
python -c "import duckdb; print('DuckDB OK')"
python -c "import xgboost; print('XGBoost OK')"
\`\`\`

### 2.5 Conda Environment Contents

The \`conda_env.yaml\` includes:

- **Core:** numpy<2.0, pandas, scikit-learn, matplotlib, seaborn
- **Time Series:** statsmodels, prophet
- **ML:** lightgbm, xgboost, optuna
- **Data:** polars, duckdb
- **Deep Learning:** PyTorch with CUDA 12.1
- **Finance:** yfinance, pandas-ta
- **Utilities:** joblib, tqdm, rich, wandb

---

## 3. Directory Structure

\`\`\`
kairos_phase4/
├── CLAUDE.md                    # Project instructions for Claude Code
├── conda_env.yaml               # Conda environment definition
├── rebalance-design.md          # Rebalance system specification
├── weekly_pipeline.sh           # Weekly automation script
│
├── data/
│   └── kairos.duckdb            # Main database (~115GB)
│
├── scripts/
│   ├── run_pipeline.py          # Master orchestrator (7 phases)
│   ├── smart_data_sync.py       # API data sync with pagination
│   ├── daily_download.py        # Single-day data download
│   ├── merge_daily_download_duck.py  # Merge parquet to DuckDB
│   ├── build_feature_matrix_v2.py    # Final feature assembly
│   │
│   ├── features/                # Feature engineering (30+ scripts)
│   │   ├── price_action_features.py
│   │   ├── trend_features.py
│   │   ├── statistical_features.py
│   │   ├── volume_volatility_features.py
│   │   ├── price_shape_features.py
│   │   ├── adv_features.py
│   │   ├── vol_sizing_features.py
│   │   ├── beta_features.py
│   │   ├── generate_targets.py
│   │   ├── build_value_factors_v2.py
│   │   ├── build_quality_factors_v2.py
│   │   ├── build_momentum_factors_v2.py
│   │   ├── build_insider_factors.py
│   │   ├── institutional_factor_academic.py
│   │   ├── rebuild_feat_fundamental.py
│   │   ├── build_composite_long_v2.py
│   │   ├── build_academic_composite_factors.py
│   │   ├── build_composite_v31.py
│   │   ├── smooth_alpha_v31.py
│   │   ├── build_composite_v32b.py
│   │   ├── build_composite_v33_regime.py
│   │   ├── build_alpha_composite_v7.py
│   │   └── build_alpha_composite_v8.py
│   │
│   ├── ml/                      # ML model training and prediction
│   │   ├── generate_ml_predictions_v2.py
│   │   └── generate_ml_predictions_v2_tuned.py
│   │
│   ├── regime/                  # Market regime detection
│   │   └── regime_detector_academic.py
│   │
│   ├── production/              # Rebalancing and trade execution
│   │   ├── generate_rebalance.py
│   │   ├── execute_alpaca_rebalance.py
│   │   └── check_rebalance.py
│   │
│   └── sep_dataset/
│       ├── daily_downloads/     # Parquet staging area
│       └── feature_sets/
│           └── option_b_universe.csv
│
├── docker/
│   ├── download/                # Data download container
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── daily_download.py
│   │   └── .env                 # API key (git-ignored)
│   ├── merge/                   # Data merge container
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── merge_daily_download_duck.py
│   └── pipeline/                # Full pipeline container
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── run_pipeline.py
│       └── scripts/             # Copy of scripts directory
│
├── models/
│   ├── saved_models/            # Trained ML models
│   └── predictions/             # ML prediction outputs
│
├── outputs/
│   └── rebalance/               # Weekly rebalance outputs
│       └── {date}/
│           ├── picks.csv
│           ├── trades.csv
│           ├── portfolio_summary.json
│           └── schedule.json
│
└── logs/                        # Pipeline execution logs
\`\`\`

---

## 4. Data Sync Process

### 4.1 Smart Data Sync (Recommended)

The \`smart_data_sync.py\` script intelligently syncs SHARADAR data with pagination support.

**Tables Supported:**
| Table | DB Table | Date Field | Description |
|-------|----------|------------|-------------|
| SEP | sep_base | date | Daily stock prices |
| DAILY | daily | date | Daily fundamental ratios |
| SF1 | sf1 | lastupdated | Quarterly fundamentals |
| SF2 | sf2 | filingdate | Insider transactions |

**Basic Usage:**
\`\`\`bash
# Check and sync all tables
python scripts/smart_data_sync.py --db data/kairos.duckdb

# Check only (don't download)
python scripts/smart_data_sync.py --db data/kairos.duckdb --check-only

# Sync specific tables
python scripts/smart_data_sync.py --db data/kairos.duckdb --tables SEP DAILY

# Force download even if up to date
python scripts/smart_data_sync.py --db data/kairos.duckdb --force
\`\`\`

**CLI Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| \`--db\` | Yes | - | Path to DuckDB database |
| \`--tables\` | No | All (SEP, DAILY, SF1, SF2) | Tables to sync |
| \`--check-only\` | No | False | Only check, don't download |
| \`--force\` | No | False | Force download even if up to date |
| \`--temp-dir\` | No | \`scripts/sep_dataset/daily_downloads\` | Temp directory for downloads |

### 4.2 Single-Day Download

For downloading a specific date's data:

\`\`\`bash
# Download SEP and DAILY for a specific date
python scripts/daily_download.py --date 2026-01-15

# Download only SEP
python scripts/daily_download.py --date 2026-01-15 --tables SEP
\`\`\`

**Output:** Parquet files in \`scripts/sep_dataset/daily_downloads/\`

### 4.3 Merge Downloaded Data

After downloading, merge parquet files into DuckDB:

\`\`\`bash
python scripts/merge_daily_download_duck.py \\
    --update-golden data/kairos.duckdb \\
    --daily-dir scripts/sep_dataset/daily_downloads
\`\`\`

**Note:** The merge script deletes parquet files after successful merge.

---

## 5. Pipeline Architecture - 7 Phases

The Kairos pipeline consists of 7 sequential phases defined in \`scripts/run_pipeline.py\`.

### Phase 1: Universe & Base

**Purpose:** Create the tradable universe and base price data.

**Scripts:**
| Order | Script | Description |
|-------|--------|-------------|
| 1.1 | \`create_option_b_universe.py\` | Filter stocks by ADV (\$500K min) and price (\$2 min) |
| 1.2 | \`create_academic_base.py\` | Create sep_base_academic from universe |

**CLI Arguments for create_option_b_universe.py:**
- \`--db\` - DuckDB path (required)
- \`--min-adv\` - Minimum ADV filter (default: 500000)
- \`--min-price\` - Minimum price filter (default: 2.0)
- \`--universe-csv\` - Output universe CSV path

### Phase 2: Technical Features

**Purpose:** Generate technical indicators from price/volume data.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 2.1 | \`price_action_features.py\` | feat_price_action |
| 2.2 | \`trend_features.py\` | feat_trend |
| 2.3 | \`statistical_features.py\` | feat_statistical |
| 2.4 | \`volume_volatility_features.py\` | feat_vol_volatility |
| 2.5 | \`price_shape_features.py\` | feat_price_shape |
| 2.6 | \`adv_features.py\` | feat_adv |
| 2.7 | \`vol_sizing_features.py\` | feat_vol_sizing |
| 2.8 | \`beta_features.py\` | feat_beta |
| 2.9 | \`generate_targets.py\` | feat_targets |

### Phase 3: Fundamental Factors

**Purpose:** Build fundamental factor composites from SF1/DAILY data.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 3.1 | \`build_value_factors_v2.py\` | feat_value_v2 |
| 3.2 | \`build_quality_factors_v2.py\` | feat_quality_v2 |
| 3.3 | \`build_momentum_factors_v2.py\` | feat_momentum_v2 |
| 3.4 | \`build_insider_factors.py\` | feat_insider |
| 3.5 | \`institutional_factor_academic.py\` | feat_institutional |
| 3.6 | \`rebuild_feat_fundamental.py\` | feat_fundamental |

### Phase 4: Composites (Base)

**Purpose:** Build intermediate composite alpha signals.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 4.1 | \`build_composite_long_v2.py\` | feat_composite_long_v2 |
| 4.2 | \`build_academic_composite_factors.py\` | feat_composite_academic |
| 4.3 | \`build_composite_v31.py\` | feat_composite_v31 |
| 4.4 | \`smooth_alpha_v31.py\` | feat_composite_v31_smooth |
| 4.5 | \`build_composite_v32b.py\` | feat_composite_v32b |

### Phase 5: Regime & Final Composites

**Purpose:** Detect market regime and build final alpha signals.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 5.1 | \`regime_detector_academic.py\` | regime_history_academic |
| 5.2 | \`build_composite_v33_regime.py\` | feat_composite_v33_regime |
| 5.3 | \`build_alpha_composite_v7.py\` | feat_composite_v7 |
| 5.4 | \`build_alpha_composite_v8.py\` | alpha_composite_v8 |

**Regime States:**
- \`bull\` / \`bear\` (trend regime)
- \`normal_vol\` / \`high_vol\` (volatility regime)
- Combined: \`normal_vol_bull\`, \`high_vol_bear\`, etc.

### Phase 6: ML Predictions

**Purpose:** Generate ML-based alpha predictions.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 6.1 | \`generate_ml_predictions_v2.py\` | ml_predictions_v2 |
| 6.2 | \`generate_ml_predictions_v2_tuned.py\` | ml_predictions_v2_tuned |

### Phase 7: Feature Matrix Assembly

**Purpose:** Assemble final feature matrix for rebalancing.

**Scripts:**
| Order | Script | Output Table |
|-------|--------|--------------|
| 7.1 | \`build_feature_matrix_v2.py\` | feat_matrix_v2 |

**CLI Arguments:**
- \`--db\` - DuckDB path (required)
- \`--date\` - Target date for matrix (default: today)
- \`--universe\` - Universe CSV path

---

## 6. Running the Pipeline

### 6.1 Full Pipeline (All 7 Phases)

\`\`\`bash
python scripts/run_pipeline.py \\
    --db data/kairos.duckdb \\
    --universe scripts/sep_dataset/feature_sets/option_b_universe.csv \\
    --date 2026-01-15
\`\`\`

### 6.2 Specific Phase Only

\`\`\`bash
# Run only Phase 2 (Technical Features)
python scripts/run_pipeline.py --db data/kairos.duckdb --phase 2
\`\`\`

### 6.3 Phase Range

\`\`\`bash
# Run Phases 3 through 5
python scripts/run_pipeline.py \\
    --db data/kairos.duckdb \\
    --universe scripts/sep_dataset/feature_sets/option_b_universe.csv \\
    --date 2026-01-15 \\
    --start-phase 3 \\
    --end-phase 5
\`\`\`

### 6.4 Dry Run (Preview)

\`\`\`bash
python scripts/run_pipeline.py \\
    --db data/kairos.duckdb \\
    --universe scripts/sep_dataset/feature_sets/option_b_universe.csv \\
    --date 2026-01-15 \\
    --dry-run
\`\`\`

### 6.5 List All Scripts

\`\`\`bash
python scripts/run_pipeline.py --list
\`\`\`

### 6.6 CLI Arguments Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| \`--db\` | Yes | - | Path to DuckDB database |
| \`--universe\` | No | \`scripts/sep_dataset/feature_sets/option_b_universe.csv\` | Universe CSV |
| \`--date\` | No | Today | Date for feature matrix (YYYY-MM-DD) |
| \`--phase\` | No | - | Run only this phase (1-7) |
| \`--start-phase\` | No | 1 | Start from this phase |
| \`--end-phase\` | No | 7 | End at this phase |
| \`--dry-run\` | No | False | Show what would run |
| \`--list\` | No | False | List all pipeline scripts |

---

## 7. Weekly Automation

> **Note:** Cron automation is **planned but not yet implemented**. The weekly pipeline is currently run manually via CLI commands.

### 7.1 Current Manual Workflow

The weekly pipeline is currently executed manually using the following steps:

**Step 1: Sync latest data**
```bash
conda activate kairos-gpu
python scripts/smart_data_sync.py --db data/kairos.duckdb
```

**Step 2: Run the full pipeline**
```bash
python scripts/run_pipeline.py --db data/kairos.duckdb --date YYYY-MM-DD
```

**Step 3: Generate rebalance picks**
```bash
python scripts/production/generate_rebalance.py --db data/kairos.duckdb --date YYYY-MM-DD
```

**Step 4: Execute trades (if desired)**
```bash
python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/YYYY-MM-DD/picks.csv --execute
```

### 7.2 Weekly Pipeline Script (For Reference)

The `weekly_pipeline.sh` script exists but is designed for future Docker-based automation:

1. Downloads Monday-Friday market data for the current week
2. Merges all data into DuckDB
3. Cleans up parquet files

**Note:** This script requires Docker containers that are not yet built. See Section 10 for planned Docker deployment.

### 7.3 Future Cron Setup (Planned)

When automation is implemented, the planned schedule is:

**Cron entry (Friday 9pm):**
```cron
0 21 * * 5 /path/to/kairos_phase4/weekly_pipeline.sh
```

**Cron Format:**
- `0` - minute 0
- `21` - hour 21 (9pm)
- `*` - any day of month
- `*` - any month
- `5` - Friday (0=Sunday, 5=Friday)

### 7.4 Log Files

When running manually, monitor output directly in the terminal. Future automation will use:
`logs/weekly_pipeline_YYYYMMDD_HHMMSS.log`

---

## 8. Rebalancing System

### 8.1 Rebalance Schedule

**Rule:** Last trading day of each week (usually Friday, Thursday if Friday is a holiday).

**Check Rebalance Schedule:**
\`\`\`bash
# Check if specific date is rebalance day
python scripts/production/check_rebalance.py --date 2026-01-15

# Show next 10 rebalance dates
python scripts/production/check_rebalance.py --next 10

# Show rebalance dates in range
python scripts/production/check_rebalance.py --range 2026-01-01 2026-06-30
\`\`\`

### 8.2 Configuration Parameters

**Current Configuration (ML_CONFIG in generate_rebalance.py):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| \`top_n\` | 75 | Number of stocks to hold |
| \`target_vol\` | 0.25 (25%) | Annual volatility target |
| \`min_adv\` | \$2,000,000 | Minimum average daily volume |
| \`max_position_pct\` | 0.03 (3%) | Maximum single position |
| \`max_sector_mult\` | 2.0 | Max sector weight = 2x universe weight |
| \`lambda_tc\` | 0.5 | Turnover smoothing (0=all new, 1=no change) |
| \`max_turnover\` | 0.30 (30%) | Maximum portfolio turnover per rebalance |
| \`alpha_column\` | \`alpha_ml_v2_tuned_clf\` | Alpha signal column |
| \`vol_column\` | \`vol_blend\` | Volatility column |
| \`adv_column\` | \`adv_20\` | ADV column |

**Note:** There is also a V8_CONFIG (uses \`alpha_composite_v8\` with 20% target vol) available in the code.

### 8.3 Generate Rebalance Picks

\`\`\`bash
# Generate rebalance for specific date
python scripts/production/generate_rebalance.py \\
    --db data/kairos.duckdb \\
    --date 2026-01-15

# Generate with prior holdings (for turnover smoothing)
python scripts/production/generate_rebalance.py \\
    --db data/kairos.duckdb \\
    --date 2026-01-15 \\
    --prior-holdings outputs/rebalance/2026-01-10/picks.csv

# Generate with custom portfolio value
python scripts/production/generate_rebalance.py \\
    --db data/kairos.duckdb \\
    --date 2026-01-15 \\
    --portfolio-value 500000

# Force generate even if not rebalance day
python scripts/production/generate_rebalance.py \\
    --db data/kairos.duckdb \\
    --date 2026-01-15 \\
    --force

# Check only (don't generate)
python scripts/production/generate_rebalance.py \\
    --db data/kairos.duckdb \\
    --date 2026-01-15 \\
    --check-only
\`\`\`

**CLI Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| \`--db\` | Yes | - | Path to DuckDB database |
| \`--date\` | No | Latest in DB | Target date (YYYY-MM-DD) |
| \`--output-dir\` | No | \`outputs/rebalance\` | Output directory |
| \`--prior-holdings\` | No | - | Prior picks.csv for turnover smoothing |
| \`--portfolio-value\` | No | 100,000 | Portfolio value in \$ |
| \`--check-only\` | No | False | Only check if rebalance day |
| \`--force\` | No | False | Generate even if not rebalance day |

### 8.4 Output Artifacts

Generated in \`outputs/rebalance/{date}/\`:

**picks.csv:**
\`\`\`csv
rank,ticker,weight,sector,alpha_z,vol_blend,adv_20,price,target_value,shares
1,AAPL,0.0285,Technology,2.34,0.18,125000000,195.50,2850.00,14
2,MSFT,0.0274,Technology,2.21,0.16,98000000,380.25,2740.00,7
...
\`\`\`

**trades.csv:**
\`\`\`csv
ticker,action,trade_value,target_value,current_value,price,shares
NVDA,BUY,6250.00,6250.00,0.00,125.00,50
META,SELL,15000.00,0.00,15000.00,500.00,30
...
\`\`\`

**portfolio_summary.json:**
\`\`\`json
{
  "generated_at": "2026-01-15T08:00:00",
  "rebalance_date": "2026-01-15",
  "is_rebalance_day": true,
  "next_rebalance": "2026-01-22",
  "regime": {
    "date": "2026-01-15",
    "regime": "normal_vol_bull",
    "vol_regime": "normal_vol",
    "trend_regime": "bull"
  },
  "portfolio": {
    "n_positions": 75,
    "top_n_target": 75,
    "total_weight": 1.0,
    "top_5_weight": 0.138,
    "sector_weights": {...}
  },
  "parameters": {
    "alpha_column": "alpha_ml_v2_tuned_clf",
    "target_vol": 0.25,
    "min_adv": 2000000,
    "max_position_pct": 0.03,
    "rebalance_rule": "Last trading day of each week"
  },
  "data_freshness": {...}
}
\`\`\`

**schedule.json:**
\`\`\`json
{
  "generated_at": "2026-01-15T08:00:00",
  "rule": "Last trading day of each week",
  "next_rebalances": ["2026-01-22", "2026-01-29", ...]
}
\`\`\`

### 8.5 Risk4 Methodology

The weight calculation follows the Risk4 methodology:

1. **Z-score Alpha:** Cross-sectionally standardize alpha, clip at +/-3
2. **Select Top N:** Pick top 75 stocks by alpha_z
3. **Base Weights:** weight = alpha_z / vol_blend
4. **Position Cap:** Maximum 3% per position
5. **Sector Cap:** Maximum 2x universe sector weight (hard cap at 40%)
6. **Turnover Smoothing:** Blend with prior weights if provided
7. **Normalize:** Final weights sum to 1.0

---

## 9. Trade Execution (Alpaca)

### 9.1 Overview

The \`execute_alpaca_rebalance.py\` script executes rebalance picks via the Alpaca API.

**Order Types:**
- **Paper Trading:** Defaults to intraday market orders (MOO simulation is unreliable)
- **Live Trading:** Defaults to Market-on-Open (MOO) orders

### 9.2 Preview Orders

\`\`\`bash
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --preview
\`\`\`

### 9.3 Execute Orders

\`\`\`bash
# Execute (auto-detects paper vs live)
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --execute

# Execute with custom portfolio value
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --portfolio-value 100000 \\
    --execute

# Execute with disaster protection (skip >5% gap)
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --execute \\
    --max-gap 5
\`\`\`

### 9.4 Force Specific Order Types

\`\`\`bash
# Force intraday market orders
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --execute \\
    --intraday

# Force MOO orders on paper trading (for testing)
python scripts/production/execute_alpaca_rebalance.py \\
    --picks outputs/rebalance/2026-01-15/picks.csv \\
    --execute \\
    --force-moo
\`\`\`

### 9.5 CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| \`--picks\` | Yes | - | Path to picks.csv |
| \`--trades\` | No | - | Path to trades.csv (optional) |
| \`--portfolio-value\` | No | Account equity | Override portfolio value |
| \`--preview\` | No | - | Preview orders without executing |
| \`--execute\` | No | - | Actually submit orders |
| \`--dry-run\` | No | False | Go through motions but don't submit |
| \`--intraday\` | No | False | Force immediate market orders |
| \`--force-moo\` | No | False | Force MOO even on paper trading |
| \`--max-gap\` | No | None | Skip orders if price gapped more than X% |

### 9.6 Output

Order results saved to: \`outputs/rebalance/{date}/alpaca_orders_{timestamp}.csv\`

---

## 10. Docker Deployment

> **Status: PLANNED - NOT YET IMPLEMENTED**
>
> The Docker containerization described in this section is planned for future implementation. The directory structure and Dockerfiles exist in the `docker/` directory, but the containers have not been built or tested. Currently, all pipeline operations should be run directly via the CLI as described in Section 6.

### 10.1 Planned Architecture

Three Docker containers are planned:
- **kairos-download** - Data download from SHARADAR API
- **kairos-merge** - Merge parquet files into DuckDB
- **kairos-pipeline** - Run the 7-phase ML pipeline

### 10.2 Download Container (Planned)

**Build (when ready):**
```bash
cp scripts/daily_download.py docker/download/
cd docker/download
docker build -t kairos-download:v1 .
```

**Run (when ready):**
```bash
docker run \
    --env-file docker/download/.env \
    --user $(id -u):$(id -g) \
    -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/app/scripts/sep_dataset/daily_downloads \
    kairos-download:v1 \
    --date 2026-01-15
```

### 10.3 Merge Container (Planned)

**Build (when ready):**
```bash
cp scripts/merge_daily_download_duck.py docker/merge/
cd docker/merge
docker build -t kairos-merge:v1 .
```

**Run (when ready):**
```bash
docker run \
    --user $(id -u):$(id -g) \
    -v /path/to/kairos_phase4/data:/data \
    -v /path/to/kairos_phase4/scripts/sep_dataset/daily_downloads:/downloads \
    kairos-merge:v1 \
    --update-golden /data/kairos.duckdb \
    --daily-dir /downloads
```

### 10.4 Pipeline Container (Planned)

**Build (when ready):**
```bash
cp -r scripts docker/pipeline/
cd docker/pipeline
docker build -t kairos-pipeline:v1 .
```

**Run Full Pipeline (when ready):**
```bash
docker run \
    --user $(id -u):$(id -g) \
    -v /path/to/kairos_phase4/data:/data \
    -v /path/to/kairos_phase4/scripts/sep_dataset/feature_sets:/features \
    kairos-pipeline:v1 \
    run_pipeline.py \
    --db /data/kairos.duckdb \
    --universe /features/option_b_universe.csv \
    --date 2026-01-15
```

### 10.5 Volume Mounts Reference (Planned)

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `data/` | `/data` | DuckDB database |
| `scripts/sep_dataset/daily_downloads/` | `/downloads` | Parquet staging |
| `scripts/sep_dataset/feature_sets/` | `/features` | Universe CSV |

---

## 11. Database Tables Reference

### 11.1 Input Tables

| Table | Description | Key Columns |
|-------|-------------|-------------|
| \`sep_base\` | Raw SHARADAR daily prices | ticker, date, open, high, low, close, volume |
| \`sep_base_academic\` | Filtered universe prices | ticker, date, open, high, low, close, volume |
| \`daily\` | Daily fundamental ratios | ticker, date, pe, pb, ps, ev/ebitda |
| \`sf1\` | Quarterly fundamentals | ticker, datekey, reportperiod, revenue, netinc |
| \`sf2\` | Insider transactions | ticker, filingdate, transactionshares |
| \`tickers\` | Ticker metadata | ticker, sector, industry, table |

### 11.2 Feature Tables

| Table | Phase | Description |
|-------|-------|-------------|
| \`feat_price_action\` | 2 | Price action indicators |
| \`feat_trend\` | 2 | Trend indicators |
| \`feat_statistical\` | 2 | Statistical features |
| \`feat_vol_volatility\` | 2 | Volume/volatility features |
| \`feat_price_shape\` | 2 | Price shape features |
| \`feat_adv\` | 2 | Average daily volume |
| \`feat_vol_sizing\` | 2 | Volatility for position sizing |
| \`feat_beta\` | 2 | Beta features |
| \`feat_targets\` | 2 | Forward returns (targets) |
| \`feat_value_v2\` | 3 | Value factor composite |
| \`feat_quality_v2\` | 3 | Quality factor composite |
| \`feat_momentum_v2\` | 3 | Momentum factor composite |
| \`feat_insider\` | 3 | Insider trading signals |
| \`feat_institutional\` | 3 | Institutional ownership |
| \`feat_composite_long_v2\` | 4 | Long composite |
| \`feat_composite_academic\` | 4 | Academic composite |
| \`feat_composite_v31\` | 4 | V31 composite |
| \`feat_composite_v32b\` | 4 | V32b composite |
| \`feat_composite_v33_regime\` | 5 | Regime-aware composite |
| \`alpha_composite_v8\` | 5 | Final alpha signal |

### 11.3 Output Tables

| Table | Description |
|-------|-------------|
| \`regime_history_academic\` | Bull/bear + volatility regime state |
| \`feat_matrix_v2\` | Final 100+ column feature set for rebalancing |
| \`ml_predictions_v2\` | ML model predictions |
| \`ml_predictions_v2_tuned\` | Tuned ML model predictions |

---

## 12. Common Commands Quick Reference

### Environment
\`\`\`bash
conda activate kairos-gpu
\`\`\`

### Data Sync
\`\`\`bash
# Full sync
python scripts/smart_data_sync.py --db data/kairos.duckdb

# Check only
python scripts/smart_data_sync.py --db data/kairos.duckdb --check-only
\`\`\`

### Run Pipeline
\`\`\`bash
# Full pipeline
python scripts/run_pipeline.py --db data/kairos.duckdb --date 2026-01-15

# Single phase
python scripts/run_pipeline.py --db data/kairos.duckdb --phase 5

# List structure
python scripts/run_pipeline.py --list
\`\`\`

### Rebalancing
\`\`\`bash
# Check schedule
python scripts/production/check_rebalance.py --date 2026-01-15

# Generate picks
python scripts/production/generate_rebalance.py --db data/kairos.duckdb --date 2026-01-15

# Preview execution
python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2026-01-15/picks.csv --preview

# Execute
python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2026-01-15/picks.csv --execute
\`\`\`

### Verify Data
\`\`\`bash
python3 -c "
import duckdb
con = duckdb.connect('data/kairos.duckdb', read_only=True)
print('Last sep_base date:', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0])
print('Last feat_matrix date:', con.execute('SELECT MAX(date) FROM feat_matrix_v2').fetchone()[0])
con.close()
"
\`\`\`

---

## 13. Troubleshooting

### 13.1 Data Sync Issues

**"Environment variable not found" error:**
\`\`\`bash
export NASDAQ_DATA_LINK_API_KEY="your-key"
\`\`\`

**"Permission denied" on output directory:**
\`\`\`bash
mkdir -p scripts/sep_dataset/daily_downloads
chmod 755 scripts/sep_dataset/daily_downloads
\`\`\`

### 13.2 Pipeline Issues

**"Script not found" warnings:**
- The pipeline continues even if optional scripts are missing
- Ensure all scripts are copied to the correct directories

**"Table does not exist" errors:**
- Run phases in order (1 through 7) for a clean build
- Phase dependencies require earlier phases to complete

**Memory issues:**
- Ensure 16GB+ RAM available
- Close other applications using DuckDB
- Consider running phases individually

### 13.3 Docker Issues (For Future Use)

> **Note:** Docker containers are planned but not yet implemented. This section is provided for future reference.

**"Permission denied" from cron:**
```bash
# Ensure user is in docker group
groups  # Should show 'docker'
sudo usermod -aG docker $USER
# Log out and back in
```

**Docker not found from cron:**
```bash
# Use full path in weekly_pipeline.sh
which docker  # Usually /usr/bin/docker
# Edit script to use full path
```

**Database locking:**
- Ensure no other process has DuckDB open when running merge

### 13.4 Alpaca Issues

**"Account not active" error:**
- Check API credentials
- Verify account status in Alpaca dashboard

**Orders failing:**
- Check buying power vs order total
- Verify symbols are tradable on Alpaca

### 13.5 Check Cron Logs (For Future Use)

> **Note:** Cron automation is planned but not yet implemented. This section is provided for future reference.

```bash
grep CRON /var/log/syslog | tail -20
```

---

## 14. Known Issues

### 14.1 Alpaca Paper Trading MOO Orders

Alpaca's paper trading environment does not properly simulate Market-on-Open (MOO) orders. Approximately 27% of MOO orders expire without filling due to simulation bugs.

**Workaround:** The \`execute_alpaca_rebalance.py\` script automatically uses intraday market orders for paper trading. Use \`--force-moo\` to test MOO behavior.

**Reference:** https://forum.alpaca.markets/t/accurate-opg-and-cls-prices-for-paper-trading/3762

### 14.2 DuckDB Memory Requirements

DuckDB operations are memory-intensive. The ~115GB database requires sufficient RAM for large queries.

**Recommendation:** 16GB minimum, 32GB recommended.

### 14.3 Automation Not Yet Implemented

Docker containers and cron-based automation are planned but not yet implemented. The system is currently operated manually via CLI commands. See:
- Section 6 for running the pipeline manually
- Section 7 for the planned automation approach
- Section 10 for planned Docker deployment

### 14.4 Documentation Discrepancies

During the creation of this guide, the following discrepancies were noted between existing documentation and actual code:

1. **Rebalance Schedule:** \`rebalance-design.md\` describes a fixed-anchor schedule (every 5 trading days), but \`generate_rebalance.py\` uses "last trading day of each week."

2. **Alpha Signal:** Current code defaults to \`ML_CONFIG\` using \`alpha_ml_v2_tuned_clf\`, not the V8 alpha signal mentioned in some docs.

3. **Target Volatility:** ML_CONFIG uses 25% target vol, not the 20% mentioned in older documentation.

4. **Pipeline Phases:** The docstring in \`run_pipeline.py\` references 5 phases, but the actual code defines 7 phases.

---

## Appendix A: NYSE Holiday Calendar (2024-2026)

\`\`\`
2024: Jan 1, Jan 15, Feb 19, Mar 29, May 27, Jun 19, Jul 4, Sep 2, Nov 28, Dec 25
2025: Jan 1, Jan 20, Feb 17, Apr 18, May 26, Jun 19, Jul 4, Sep 1, Nov 27, Dec 25
2026: Jan 1, Jan 19, Feb 16, Apr 3, May 25, Jun 19, Jul 3, Sep 7, Nov 26, Dec 25
\`\`\`

---

## Appendix B: Key Files Quick Reference

| File | Purpose |
|------|---------|
| \`scripts/run_pipeline.py\` | Master orchestrator |
| \`scripts/smart_data_sync.py\` | API data sync |
| \`scripts/production/generate_rebalance.py\` | Risk4 rebalancing |
| \`scripts/production/execute_alpaca_rebalance.py\` | Trade execution |
| \`scripts/production/check_rebalance.py\` | Schedule checker |
| \`scripts/build_feature_matrix_v2.py\` | Final feature assembly |
| \`scripts/features/build_alpha_composite_v8.py\` | Best alpha signal |
| \`weekly_pipeline.sh\` | Weekly automation |
| \`conda_env.yaml\` | Environment definition |
| \`rebalance-design.md\` | System specification |

---

*End of Kairos Pipeline Complete Guide*
