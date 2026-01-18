# Weekly Pipeline & Rebalance Agent

**Version:** 2.0
**Last Updated:** January 2026
**Author:** Kairos Quant Engineering

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Weekly Data Pipeline](#3-weekly-data-pipeline)
4. [Rebalance Agent](#4-rebalance-agent)
5. [Execution Engine](#5-execution-engine)
6. [Operational Runbook](#6-operational-runbook)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Overview

The Kairos Weekly Pipeline & Rebalance Agent is an end-to-end automated system for quantitative portfolio management. It handles three core functions:

| Component | Function | Schedule |
|-----------|----------|----------|
| **Data Pipeline** | Download market data, merge into DuckDB, generate features | Friday 9pm ET |
| **Rebalance Agent** | Select stocks, calculate weights, generate trades | Last trading day of week |
| **Execution Engine** | Submit orders via Alpaca API | After rebalance approval |

### Key Characteristics

- **75-stock portfolio** using Risk4/ML methodology
- **25% target volatility** with position-level risk controls
- **30% max turnover** per rebalance to control costs
- **Automated paper trading** with manual approval for live trading
- **NYSE calendar aware** for accurate trading day calculations

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        WEEKLY PIPELINE & REBALANCE FLOW                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  FRIDAY 9PM  │───▶│  SATURDAY    │───▶│  REBALANCE   │───▶│  EXECUTION   │  │
│  │  Data Sync   │    │  Features    │    │  Day Check   │    │  (if ready)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼          │
│    Download SEP        Build 50+          Generate picks       Submit via      │
│    Mon-Fri data        features           & trade list         Alpaca API      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           DuckDB Database                                │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │  sep_base → feat_matrix_v2 → alpha_ml_v2_tuned_clf → rebalance_picks   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interactions

```
                    ┌─────────────────────┐
                    │   Trading Calendar  │
                    │   (NYSE holidays)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  weekly_pipeline│ │ check_rebalance │ │generate_rebalance│
    │     _full.sh    │ │      .py        │ │      .py        │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   DuckDB        │ │  schedule.json  │ │   picks.csv     │
    │   Database      │ │                 │ │   trades.csv    │
    └─────────────────┘ └─────────────────┘ └────────┬────────┘
                                                     │
                                                     ▼
                                           ┌─────────────────┐
                                           │execute_alpaca_  │
                                           │  rebalance.py   │
                                           └─────────────────┘
```

### 2.3 Directory Structure

```
kairos-modeling/
├── Weekly Pipeline Scripts
│   ├── weekly_pipeline.sh           # Basic: download + merge
│   ├── weekly_pipeline_full.sh      # Full: download + merge + features
│   └── daily_update_pipeline.sh     # Daily incremental updates
│
├── Rebalance Agent Scripts
│   └── scripts/production/
│       ├── generate_rebalance.py    # Stock selection & weights
│       ├── check_rebalance.py       # Schedule determination
│       ├── execute_alpaca_rebalance.py  # Order submission
│       └── track_rebalance_performance.py  # Performance monitoring
│
├── Portfolio Builders
│   └── scripts/portfolio/
│       ├── prepare_weekly_state.py  # Data readiness check
│       ├── run_weekly_primary.py    # Primary long-only portfolio
│       ├── run_weekly_regime.py     # Regime-aware adjustments
│       └── run_weekly_hedge.py      # Hedge portfolio construction
│
├── Docker Containers
│   └── docker/
│       ├── download/                # SHARADAR data download
│       ├── merge/                   # DuckDB merge operations
│       └── pipeline/                # Feature generation (Phases 1-5)
│
├── Output Artifacts
│   └── outputs/
│       └── rebalance/
│           └── YYYY-MM-DD/          # Date-stamped outputs
│               ├── picks.csv
│               ├── trades.csv
│               ├── portfolio_summary.json
│               └── schedule.json
│
└── Configuration & Data
    ├── data/kairos.duckdb           # Main database
    └── conda_env.yaml               # Environment specification
```

---

## 3. Weekly Data Pipeline

### 3.1 Pipeline Phases

The full weekly pipeline executes in four phases:

| Phase | Script/Container | Description | Duration |
|-------|-----------------|-------------|----------|
| **A** | `docker/download` | Download Mon-Fri SHARADAR SEP data | ~5 min |
| **B** | `docker/merge` | Merge parquet files into DuckDB | ~10 min |
| **C** | Cleanup | Remove temporary parquet files | ~1 min |
| **D** | `docker/pipeline` | Generate all features (Phases 1-5) | ~45 min |

### 3.2 Feature Generation (Phase D)

Phase D runs the full feature pipeline in sequence:

```
Phase 1: Universe & Base
    └── filter_common_duck.py (liquid stocks, sector info)

Phase 2: Technical Features
    └── price_features, trend_features, volume_features
    └── momentum, volatility, shape indicators

Phase 3: Fundamental Features
    └── SHARADAR fundamentals processing
    └── Quality, value, growth factors

Phase 4: Composite Signals
    └── alpha_composite_v8 (rule-based)
    └── alpha_composite_v33_regime (regime-aware)

Phase 5: Regime Detection
    └── Market regime classification
    └── Volatility regime indicators
```

### 3.3 Running the Pipeline

**Automated (Cron):**
```bash
# Every Friday at 9pm ET
0 21 * * 5 /home/user/kairos-modeling/weekly_pipeline_full.sh
```

**Manual Execution:**
```bash
# Full pipeline (download + features)
cd /home/user/kairos-modeling
./weekly_pipeline_full.sh

# Data only (no feature regeneration)
./weekly_pipeline.sh
```

**Daily Updates:**
```bash
# Run daily to keep data current
./daily_update_pipeline.sh
```

### 3.4 Pipeline Outputs

After successful completion:

| Output | Location | Description |
|--------|----------|-------------|
| Market data | `sep_base` table | Raw OHLCV + fundamentals |
| Feature matrix | `feat_matrix_v2` table | All computed features |
| ML predictions | `alpha_ml_v2_tuned_clf` column | ML model scores |
| Logs | `logs/weekly_pipeline_*.log` | Execution logs |

---

## 4. Rebalance Agent

### 4.1 Rebalance Schedule

The agent rebalances on the **last trading day of each week** (typically Friday).

**Schedule Logic:**
- Uses NYSE trading calendar with full holiday list
- Skips weekends and market holidays
- Half-days are treated as full trading days
- If Friday is a holiday, rebalances Thursday

**Check if today is a rebalance day:**
```bash
python scripts/production/check_rebalance.py check --date 2026-01-17
```

**Get next N rebalance dates:**
```bash
python scripts/production/check_rebalance.py next --count 10
```

### 4.2 Stock Selection (Risk4/ML Methodology)

The rebalance agent selects stocks using this algorithm:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STOCK SELECTION ALGORITHM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: LOAD DATA                                              │
│    └── Query feat_matrix_v2 for latest date                    │
│    └── Join sector info from tickers table                     │
│    └── Require: alpha_ml_v2_tuned_clf, vol_blend, adv_20       │
│                                                                 │
│  Step 2: FILTER UNIVERSE                                        │
│    └── Drop missing sector or vol_blend                        │
│    └── ADV filter: adv_20 >= $2M                               │
│    └── Typically yields 500-800 eligible stocks                │
│                                                                 │
│  Step 3: SCORE & RANK                                           │
│    └── Z-score alpha cross-sectionally                         │
│    └── Clip at ±3.0 standard deviations                        │
│    └── Rank by alpha_z descending                              │
│    └── Select top 75 stocks                                    │
│                                                                 │
│  Step 4: CALCULATE WEIGHTS                                      │
│    └── Base weight = alpha_z / vol_blend                       │
│    └── Normalize to sum to 1.0                                 │
│    └── Apply position cap (max 3%)                             │
│    └── Apply sector cap (max 2× universe weight)               │
│    └── Re-normalize                                            │
│                                                                 │
│  Step 5: TURNOVER SMOOTHING                                     │
│    └── Blend with prior: w = λ·w_prior + (1-λ)·w_target        │
│    └── λ = 0.5 (half prior, half new)                          │
│    └── Cap total turnover at 30%                               │
│                                                                 │
│  Step 6: VOLATILITY TARGETING                                   │
│    └── Estimate portfolio volatility                           │
│    └── Scale weights to achieve 25% annual vol target          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Running the Rebalance Agent

**Generate picks for today:**
```bash
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --output-dir outputs/rebalance/$(date +%Y-%m-%d)
```

**Generate picks for a specific date:**
```bash
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2026-01-17 \
    --output-dir outputs/rebalance/2026-01-17
```

**With prior holdings for turnover smoothing:**
```bash
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2026-01-17 \
    --prior-holdings outputs/rebalance/2026-01-10/picks.csv \
    --output-dir outputs/rebalance/2026-01-17
```

### 4.4 Rebalance Output Artifacts

**picks.csv** - Full stock list with weights:
```csv
ticker,weight,target_shares,sector,alpha_z,vol_blend,adv_20,rank
NVDA,0.0295,125,Technology,2.45,0.22,185000000,1
AAPL,0.0285,450,Technology,2.34,0.18,125000000,2
MSFT,0.0274,220,Technology,2.21,0.16,98000000,3
...
```

**trades.csv** - Actionable trade orders:
```csv
ticker,action,shares,estimated_value,price,reason
NVDA,BUY,50,6250.00,125.00,new_position
AAPL,BUY,25,4875.00,195.00,increase_weight
META,SELL,30,15000.00,500.00,decrease_weight
TSLA,SELL,100,24800.00,248.00,exit_position
```

**portfolio_summary.json** - Summary metrics:
```json
{
    "date": "2026-01-17",
    "is_rebalance_day": true,
    "next_rebalance": "2026-01-24",
    "portfolio": {
        "n_positions": 75,
        "total_weight": 1.0,
        "top_5_weight": 0.138,
        "estimated_vol": 0.25
    },
    "metrics": {
        "turnover_vs_prior": 0.18,
        "avg_adv": 45000000
    }
}
```

**schedule.json** - Upcoming rebalance dates:
```json
{
    "generated_at": "2026-01-17T08:00:00",
    "next_10_rebalances": [
        "2026-01-24", "2026-01-31", "2026-02-07",
        "2026-02-14", "2026-02-21", "2026-02-28",
        "2026-03-07", "2026-03-14", "2026-03-21",
        "2026-03-28"
    ]
}
```

---

## 5. Execution Engine

### 5.1 Alpaca API Integration

The execution engine submits orders through the Alpaca trading API with automatic detection of paper vs. live trading.

**Key Features:**
- **Auto-detection**: Detects paper trading via API URL
- **Paper trading**: Uses intraday market orders (MOO unreliable in paper)
- **Live trading**: Uses Market-on-Open (MOO) orders
- **Disaster protection**: `--max-gap` flag prevents large unexpected trades

### 5.2 Order Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PREVIEW MODE (default)                                      │
│     └── Load picks.csv                                         │
│     └── Calculate required trades                              │
│     └── Display summary (no orders submitted)                  │
│                                                                 │
│  2. EXECUTE MODE (--execute flag)                               │
│     └── Detect paper vs live trading                           │
│     └── Paper: Submit intraday market orders                   │
│     └── Live: Submit MOO orders before market open             │
│     └── Log all submitted orders                               │
│                                                                 │
│  3. VERIFICATION                                                │
│     └── Check order fills after market open                    │
│     └── Log any partial fills or failures                      │
│     └── Generate execution report                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Running the Execution Engine

**Preview trades (no execution):**
```bash
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-17/picks.csv
```

**Execute trades (paper trading):**
```bash
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-17/picks.csv \
    --execute
```

**Execute with portfolio value override:**
```bash
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-17/picks.csv \
    --execute \
    --portfolio-value 100000
```

**With disaster protection (max 5% gap per position):**
```bash
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-17/picks.csv \
    --execute \
    --max-gap 0.05
```

### 5.4 Environment Setup for Alpaca

Required environment variables:
```bash
export APCA_API_KEY_ID="your_api_key"
export APCA_API_SECRET_KEY="your_secret_key"

# Paper trading (default)
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"

# Live trading
export APCA_API_BASE_URL="https://api.alpaca.markets"
```

---

## 6. Operational Runbook

### 6.1 Weekly Schedule

| Day | Time | Action | Automated? |
|-----|------|--------|------------|
| **Friday** | 9:00 PM | Run weekly_pipeline_full.sh | Yes (cron) |
| **Saturday** | Morning | Verify pipeline completion | Manual check |
| **Last Trading Day** | 7:00 AM | Check if rebalance day | Manual/Script |
| **Last Trading Day** | 7:30 AM | Generate rebalance picks | Manual trigger |
| **Last Trading Day** | 8:00 AM | Review picks, approve | Manual review |
| **Last Trading Day** | 9:00 AM | Execute trades (paper) | Manual trigger |
| **Last Trading Day** | 9:30 AM | Verify order fills | Manual check |

### 6.2 Pre-Rebalance Checklist

Before generating picks, verify:

- [ ] Pipeline completed successfully (check logs)
- [ ] `feat_matrix_v2` has current date
- [ ] ML predictions (`alpha_ml_v2_tuned_clf`) are present
- [ ] At least 500 stocks pass ADV filter
- [ ] All 11 GICS sectors represented

**Run data readiness check:**
```bash
python scripts/portfolio/prepare_weekly_state.py
```

### 6.3 Post-Rebalance Checklist

After execution:

- [ ] All orders filled (check Alpaca dashboard)
- [ ] No partial fills or rejections
- [ ] Portfolio matches picks.csv (within tolerance)
- [ ] Log execution report
- [ ] Archive outputs to dated folder

**Track performance:**
```bash
python scripts/production/track_rebalance_performance.py \
    --from-date 2026-01-10 \
    --to-date 2026-01-17
```

### 6.4 Monitoring Alerts

| Condition | Severity | Action |
|-----------|----------|--------|
| Data >1 day stale | **CRITICAL** | Block rebalance, investigate |
| <50 stocks pass filter | **CRITICAL** | Block rebalance, check data |
| Turnover >50% | **WARNING** | Review before proceeding |
| Single position >5% | **WARNING** | Check cap logic |
| Order rejection | **HIGH** | Check account, retry |
| Partial fill | **MEDIUM** | May need manual completion |

---

## 7. Configuration Reference

### 7.1 Risk4/ML Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_n` | 75 | Number of stocks in portfolio |
| `target_vol` | 0.25 | Target annual volatility (25%) |
| `min_adv` | $2,000,000 | Minimum avg daily volume |
| `max_position_pct` | 0.03 | Maximum weight per stock (3%) |
| `max_sector_mult` | 2.0 | Max sector weight = 2× universe |
| `lambda_tc` | 0.5 | Turnover smoothing factor |
| `max_turnover` | 0.30 | Maximum turnover per rebalance |
| `alpha_column` | `alpha_ml_v2_tuned_clf` | ML signal column |
| `vol_column` | `vol_blend` | Volatility estimate column |
| `adv_column` | `adv_20` | ADV filter column |

### 7.2 Trading Calendar Holidays (2024-2026)

```python
NYSE_HOLIDAYS = [
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
]
```

### 7.3 Environment Requirements

**Conda Environment:** `kairos-gpu`

Key dependencies:
```yaml
- python=3.11
- pandas>=2.0
- polars>=0.19
- duckdb>=0.9
- numpy>=1.24
- scikit-learn>=1.3
- lightgbm>=4.0
- alpaca-trade-api>=3.0
- exchange_calendars>=4.0
```

Activate environment:
```bash
conda activate kairos-gpu
```

---

## 8. Troubleshooting

### 8.1 Pipeline Issues

**Problem:** Pipeline fails with Docker permission error
```
Solution: Ensure user is in docker group
$ sudo usermod -aG docker $USER
$ newgrp docker
```

**Problem:** Download fails with API error
```
Solution: Check NASDAQ API key in docker/download/.env
$ cat docker/download/.env
NASDAQ_API_KEY=your_key_here
```

**Problem:** Merge fails with DuckDB lock
```
Solution: Ensure no other process has database open
$ fuser data/kairos.duckdb  # Check for locks
$ kill <pid>                 # Kill if necessary
```

### 8.2 Rebalance Agent Issues

**Problem:** No stocks pass ADV filter
```
Solution: Check data freshness
$ python -c "import duckdb; print(duckdb.connect('data/kairos.duckdb').execute('SELECT MAX(date) FROM feat_matrix_v2').fetchone())"
```

**Problem:** Alpha column has all NULLs
```
Solution: Regenerate ML predictions
$ python backfill_ml_predictions.py
```

**Problem:** Turnover exceeds 30% cap
```
Solution: Check prior holdings file, may need fresh start
$ python scripts/production/generate_rebalance.py --no-prior
```

### 8.3 Execution Issues

**Problem:** MOO orders expire unfilled (paper trading)
```
Solution: This is expected for paper trading. The system auto-detects
paper trading and uses intraday orders instead. Verify auto-detection:
$ python scripts/production/execute_alpaca_rebalance.py --picks picks.csv
# Should show: "Paper trading detected, using intraday orders"
```

**Problem:** Insufficient buying power
```
Solution: Check account balance and pending orders
$ python -c "from alpaca_trade_api import REST; api=REST(); print(api.get_account())"
```

**Problem:** Order rejected for non-tradable symbol
```
Solution: Symbol may be halted or delisted. Remove from picks.csv
and regenerate trades, or manually adjust.
```

### 8.4 Common Log Locations

| Log | Location |
|-----|----------|
| Weekly pipeline | `logs/weekly_pipeline_*.log` |
| Docker containers | `docker logs <container_id>` |
| Alpaca orders | Alpaca dashboard / API |
| Cron jobs | `/var/log/syslog` (grep CRON) |

### 8.5 Emergency Procedures

**Cancel all pending orders:**
```bash
python -c "
from alpaca_trade_api import REST
api = REST()
for order in api.list_orders(status='open'):
    api.cancel_order(order.id)
    print(f'Cancelled: {order.symbol} {order.side} {order.qty}')
"
```

**Liquidate all positions (EMERGENCY ONLY):**
```bash
python -c "
from alpaca_trade_api import REST
api = REST()
api.close_all_positions()
print('All positions closed')
"
```

---

## Quick Reference Commands

```bash
# Check pipeline status
tail -f logs/weekly_pipeline_*.log

# Check if today is rebalance day
python scripts/production/check_rebalance.py check --date $(date +%Y-%m-%d)

# Get next 5 rebalance dates
python scripts/production/check_rebalance.py next --count 5

# Generate rebalance picks
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --output-dir outputs/rebalance/$(date +%Y-%m-%d)

# Preview execution
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/$(date +%Y-%m-%d)/picks.csv

# Execute trades
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/$(date +%Y-%m-%d)/picks.csv \
    --execute

# Check portfolio performance
python scripts/production/track_rebalance_performance.py
```

---

*End of Weekly Pipeline & Rebalance Agent Documentation*
