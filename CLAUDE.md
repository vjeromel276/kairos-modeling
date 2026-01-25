# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kairos Phase 4 is a quantitative trading system implementing a 7-phase ML pipeline for stock selection and weekly portfolio rebalancing. It uses the "Risk4" methodology combined with machine learning predictions to generate stock picks, executed via Alpaca API.

**Core workflow:** Download SHARADAR SEP data -> Build features -> Generate alpha signals -> Detect market regime -> Produce weekly rebalance recommendations -> Execute trades.

## Technology Stack

- **Python 3.11** with conda environment (`conda_env.yaml`)
- **DuckDB** (~115GB database at `data/kairos.duckdb`)
- **ML:** XGBoost, Optuna for hyperparameter tuning
- **Data Processing:** Pandas, Polars, NumPy
- **Data Source:** SHARADAR SEP API (via Nasdaq Data Link)
- **Broker:** Alpaca API (paper and live trading)
- **Execution:** Manual CLI (automation via Docker/cron planned but not yet implemented)

## Quick Start - Weekly Rebalance

For step-by-step weekly rebalance instructions, see: **`docs/WEEKLY_REBALANCE_QUICKREF.md`**

## Common Commands

### Environment Setup (IMPORTANT - Always Run First)

```bash
# Initialize and activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
```

First time setup:
```bash
conda env create -f conda_env.yaml
conda activate kairos-gpu
```

### Check Data Freshness

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && python3 -c "
import duckdb
con = duckdb.connect('data/kairos.duckdb', read_only=True)
print('sep_base:', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0])
print('feat_matrix_v2:', con.execute('SELECT MAX(date) FROM feat_matrix_v2').fetchone()[0])
print('regime_history:', con.execute('SELECT MAX(date) FROM regime_history_academic').fetchone()[0])
con.close()
"
```

### Run Full Pipeline (all 7 phases)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/run_pipeline.py --db data/kairos.duckdb --date 2026-01-15
```

### Run Specific Phase
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/run_pipeline.py --db data/kairos.duckdb --phase 5
```

### List Pipeline Structure
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/run_pipeline.py --list
```

### Data Sync (download latest SHARADAR data)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/smart_data_sync.py --db data/kairos.duckdb
```

### Weekly Rebalance Workflow

```bash
# Step 1: Check rebalance schedule
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/check_rebalance.py --date YYYY-MM-DD

# Step 2: Generate picks (with prior holdings for turnover smoothing)
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date YYYY-MM-DD \
    --portfolio-value XXXXX \
    --prior-holdings outputs/rebalance/PRIOR-DATE/picks.csv

# Step 3: Preview orders
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv \
    --preview

# Step 4: Execute orders (type YES when prompted)
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv \
    --execute
```

## Pipeline Architecture

The system processes data through 7 sequential phases defined in `scripts/run_pipeline.py`:

| Phase | Name | Key Scripts |
|-------|------|-------------|
| 1 | Universe & Base | `create_option_b_universe.py`, `create_academic_base.py` |
| 2 | Technical Features | `price_action_features.py`, `trend_features.py`, `statistical_features.py`, etc. |
| 3 | Fundamental Factors | `build_value_factors_v2.py`, `build_quality_factors_v2.py`, `build_momentum_factors_v2.py` |
| 4 | Composites (Base) | `build_composite_long_v2.py`, `build_composite_v31.py`, `build_composite_v32b.py` |
| 5 | Regime & Final Composites | `regime_detector_academic.py`, `build_composite_v33_regime.py`, `build_alpha_composite_v8.py` |
| 6 | ML Predictions | `generate_ml_predictions_v2.py`, `generate_ml_predictions_v2_tuned.py` |
| 7 | Feature Matrix Assembly | `build_feature_matrix_v2.py` |

## Key Database Tables

**Input:**
- `sep_base_academic` - Daily OHLCV data
- `daily` - Daily fundamental ratios
- `sf1` - Quarterly fundamentals
- `tickers` - Ticker metadata including sector

**Feature Tables:**
- `feat_technical_*` - Technical indicators
- `feat_value_v2`, `feat_quality_v2`, `feat_momentum_v2` - Factor composites
- `feat_composite_v33_regime` - Regime-aware alpha
- `alpha_ml_v2_tuned_clf` - ML-based alpha signal (default for rebalancing)
- `alpha_composite_v8` - Rule-based alpha signal (legacy)

**Output:**
- `feat_matrix_v2` - Final 100+ column feature set for rebalancing
- `regime_history_academic` - Bull/bear + volatility regime state

## Rebalancing System

Key parameters (from `scripts/production/generate_rebalance.py` ML_CONFIG):
- **TOP_N:** 75 stocks
- **REBALANCE_FREQ:** Last trading day of each week (weekly rebalance)
- **TARGET_VOL:** 25% annual
- **MIN_ADV:** $2M average daily volume
- **MAX_POSITION:** 3% per stock
- **MAX_SECTOR:** 2x universe weight
- **MAX_TURNOVER:** 30% per rebalance
- **ALPHA_COLUMN:** `alpha_ml_v2_tuned_clf` (ML-based predictions)

Output artifacts in `outputs/rebalance/{date}/`:
- `picks.csv` - Stock picks with target weights
- `trades.csv` - Buy/sell orders
- `portfolio_summary.json` - Summary metrics
- `schedule.json` - Upcoming rebalance dates
- `alpaca_orders_*.csv` - Order execution results

## Alpaca Configuration

Alpaca API credentials are hardcoded as defaults in `scripts/production/execute_alpaca_rebalance.py` (lines 61-62). No environment variables needed for paper trading.

**Paper Trading Notes:**
- Auto-uses intraday market orders (MOO simulation is unreliable)
- If market is closed, orders go to "accepted" status and execute at next open
- Use `--force-moo` only for testing MOO behavior

## Directory Structure

```
scripts/
├── run_pipeline.py          # Master orchestrator
├── smart_data_sync.py       # API data sync
├── features/                # Feature engineering (30+ scripts)
├── ml/                      # ML model training and prediction
├── regime/                  # Market regime detection
├── production/              # Rebalancing and trade execution
├── backtesting/             # Validation and performance testing
└── backtest_validation_suite/

docker/                      # (Planned, not yet implemented)
├── download/                # Data download container
├── merge/                   # Data merge container
└── pipeline/                # Full pipeline container

data/
└── kairos.duckdb            # Main database (~115GB)

models/
├── saved_models/            # Trained ML models
└── predictions/             # ML prediction outputs

outputs/
└── rebalance/               # Weekly rebalance outputs

docs/
├── KAIROS_PIPELINE_GUIDE.md      # Comprehensive documentation
└── WEEKLY_REBALANCE_QUICKREF.md  # Quick reference for weekly runs
```

## Key Files to Understand First

1. `scripts/run_pipeline.py` - Core orchestration logic, defines all 7 phases
2. `scripts/production/generate_rebalance.py` - Risk4 rebalancing algorithm (see ML_CONFIG for current parameters)
3. `docs/WEEKLY_REBALANCE_QUICKREF.md` - **Quick reference for weekly rebalance runs**
4. `docs/KAIROS_PIPELINE_GUIDE.md` - Comprehensive consolidated documentation
5. `scripts/build_feature_matrix_v2.py` - Final feature assembly
6. `scripts/ml/generate_ml_predictions_v2_tuned.py` - ML alpha signal generation

## Known Issues

- Alpaca paper trading MOO (market-on-open) orders unreliable; system auto-uses intraday market orders as workaround
- DuckDB requires sufficient RAM (~16GB recommended) for large queries
- "asset is not active" errors during order execution are normal for delisted stocks - safe to ignore
