# Kairos Weekly Rebalance Quick Reference

**Last Updated:** 2026-04-19
**Based on:** Live slippage analysis (outputs/evaluation/live_slippage)

---

## Cadence — run pipeline Monday morning, not Friday night

Measured live slippage on 2026-01 through 2026-04: **~88 bps per trade** weighted mean
when running Friday night → filling Monday open. Root cause: 64-hour signal-to-execution
gap lets other quants act on the same Friday close data before your orders fill.

**New recommended cadence:**

| Step | When | Why |
|---|---|---|
| data sync + pipeline + generate_rebalance | **Monday 7-8 AM ET** | pipeline runs in <1 hour on Friday close data |
| refresh_picks_prices | **Monday 9:25 AM ET** | re-anchors picks.csv price column to live Alpaca quotes; recomputes share counts |
| execute (limit orders) | **Monday 9:30 AM ET** | fills minutes after reference price is set — matches backtest assumption |
| retry-unfilled (sells as market, buys wider limit) | **Monday 9:45 AM ET** | ensures sells clear; gives buys a second chance at tighter price |

Running Sat/Sun is fine for the pipeline step (data is already published by then), but
**do not execute orders on a Saturday-night queue** — that is what produces the 88 bps
slippage. The fixable lag is between signal anchoring and order fill.

---

## TL;DR - Copy-Paste Workflow

```bash
# Step 0: Activate environment (ALWAYS DO THIS FIRST)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Step 1: Data sync + pipeline (Monday 7-8 AM)
python scripts/smart_data_sync.py --db data/kairos.duckdb
# (re-run pipeline if needed: python scripts/run_pipeline.py --db data/kairos.duckdb)

# Step 2: Generate picks
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date YYYY-MM-DD \
    --from-alpaca \
    --prior-holdings outputs/rebalance/PRIOR-DATE/picks.csv

# Step 3: Refresh picks.csv prices right before open (Monday 9:25 AM)
python scripts/production/refresh_picks_prices.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv

# Step 4: Preview limit orders
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv \
    --order-type limit --limit-buffer-bps 30 --preview

# Step 5: Execute limit orders at/just after 9:30 open
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv \
    --order-type limit --limit-buffer-bps 30 --execute <<< "YES"

# Step 6: Retry anything that didn't fill (~9:45 AM)
# SELLs retry as market (never want to hold an exiting position);
# BUYs retry with wider buffer.
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/YYYY-MM-DD/picks.csv \
    --retry-unfilled --retry-sells-as-market \
    --order-type limit --limit-buffer-bps 60 --execute <<< "YES"
```

---

## Complete Workflow with Explanations

### Step 0: Environment Setup

**CRITICAL:** Always activate conda first. Commands will fail without this.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
```

Verify it worked:
```bash
which python
# Should show: /home/<user>/miniconda3/envs/kairos-gpu/bin/python
```

### Step 1: Check Data Freshness

Before generating picks, verify the pipeline data is current:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && python3 -c "
import duckdb
con = duckdb.connect('data/kairos.duckdb', read_only=True)
print('=== DATA FRESHNESS CHECK ===')
print('sep_base (raw prices):', con.execute('SELECT MAX(date) FROM sep_base').fetchone()[0])
print('sep_base_academic:', con.execute('SELECT MAX(date) FROM sep_base_academic').fetchone()[0])
print('feat_matrix_v2 (features):', con.execute('SELECT MAX(date) FROM feat_matrix_v2').fetchone()[0])
print('regime_history_academic:', con.execute('SELECT MAX(date) FROM regime_history_academic').fetchone()[0])
con.close()
"
```

**Expected:** All dates should match your target rebalance date (or be within 1-2 days).

### Step 2: Check Rebalance Schedule

Verify the target date is a valid rebalance day:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/check_rebalance.py --date 2026-01-23
```

**Expected output:**
```
Date: 2026-01-23 (Friday)
Is trading day: Yes
Last trading day of week: 2026-01-23 (Friday)

IS REBALANCE DAY: YES
```

To see upcoming rebalance dates:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/check_rebalance.py --next 10
```

### Step 3: Find Prior Holdings

Turnover smoothing requires the previous week's holdings:

```bash
ls -la outputs/rebalance/
```

Use the most recent date's `picks.csv` as `--prior-holdings`.

### Step 4: Generate Rebalance Picks

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2026-01-23 \
    --portfolio-value 101490 \
    --prior-holdings outputs/rebalance/2026-01-16/picks.csv
```

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--db` | Path to DuckDB database | `data/kairos.duckdb` |
| `--date` | Target rebalance date | `2026-01-23` |
| `--portfolio-value` | Current portfolio value in $ | `101490` |
| `--prior-holdings` | Previous picks.csv for turnover smoothing | `outputs/rebalance/2026-01-16/picks.csv` |
| `--force` | Generate even if not a rebalance day | (optional flag) |

**Output location:** `outputs/rebalance/YYYY-MM-DD/`

### Step 5: Review Generated Picks

Check the summary:
```bash
cat outputs/rebalance/2026-01-23/portfolio_summary.json
```

Key fields to verify:
- `is_rebalance_day`: should be `true`
- `regime`: current market regime
- `n_positions`: should be 75
- `data_freshness`: all dates should match target

Review top picks:
```bash
head -25 outputs/rebalance/2026-01-23/picks.csv
```

Review trades:
```bash
head -30 outputs/rebalance/2026-01-23/trades.csv
```

### Step 6: Preview Alpaca Orders

**Always preview before executing:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-23/picks.csv \
    --preview
```

This shows:
- Account status and buying power
- All SELL orders (exits and reductions)
- All BUY orders (new positions and increases)
- Total dollar amounts
- Any warnings about buying power

### Step 7: Execute Orders

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-23/picks.csv \
    --execute
```

When prompted, type `YES` to confirm.

**Auto-pipe confirmation (for scripting):**
```bash
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-01-23/picks.csv \
    --execute <<< "YES"
```

### Step 8: Verify Execution

Check the order results file:
```bash
ls outputs/rebalance/2026-01-23/alpaca_orders_*.csv
cat outputs/rebalance/2026-01-23/alpaca_orders_*.csv | head -20
```

Count successes and failures:
```bash
grep -c "submitted" outputs/rebalance/2026-01-23/alpaca_orders_*.csv
grep -c "failed" outputs/rebalance/2026-01-23/alpaca_orders_*.csv
```

---

## Important Notes

### Alpaca Credentials

The Alpaca API credentials are **hardcoded as defaults** in `scripts/production/execute_alpaca_rebalance.py` (lines 61-62). No environment variables needed for paper trading.

**File location:** `scripts/production/execute_alpaca_rebalance.py`
```python
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "...")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
```

### Paper Trading Order Behavior

- **Paper trading auto-uses intraday market orders** (not MOO)
- This is because Alpaca's MOO simulation is unreliable (~27% fail rate)
- If market is closed, orders go to "accepted" status and execute at next market open
- Use `--force-moo` only for testing MOO behavior

### Turnover Smoothing

The `--prior-holdings` flag is important:
- First rebalance: omit this flag (all positions are new)
- Subsequent rebalances: always include prior week's picks.csv
- This helps limit turnover to ~30% per week (configurable)

### Common Order Failures

**"asset is not active"** - Stock delisted or not tradable on Alpaca
- Example: SMLR in 2026-01-23 run
- These are logged as failed but don't affect other orders

---

## Configuration Reference

Current settings in `scripts/production/generate_rebalance.py` (ML_CONFIG):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_n` | 75 | Number of stocks to hold |
| `target_vol` | 25% | Annual volatility target |
| `min_adv` | $2M | Minimum average daily volume |
| `max_position_pct` | 3% | Maximum single position |
| `max_sector_mult` | 2.0 | Max sector = 2x universe weight |
| `lambda_tc` | 0.5 | Turnover smoothing factor |
| `max_turnover` | 30% | Max turnover per rebalance |
| `alpha_column` | `alpha_ml_v2_tuned_clf` | ML-based alpha signal |

---

## Example Run: 2026-04-10

**Run Date:** 2026-04-12 (Sunday, market closed)  
**Rebalance Date:** 2026-04-10 (Friday - last trading day of week)  
**Portfolio Value:** $94,086  
**Prior Holdings:** 2026-04-02  

**Results:**
- Regime: `high_vol_neutral`
- Positions: 75
- Orders Submitted: 123
- Orders Failed: 4 (SEE, SMLR, OS - not active; 436CVR021 - CVR instrument)

**Commands used:**
```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Generate picks
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2026-04-10 \
    --portfolio-value 94086 \
    --prior-holdings outputs/rebalance/2026-04-02/picks.csv

# Preview
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-04-10/picks.csv \
    --preview

# Execute
python scripts/production/execute_alpaca_rebalance.py \
    --picks outputs/rebalance/2026-04-10/picks.csv \
    --execute <<< "YES"
```

---

## Output Files Reference

All outputs are in `outputs/rebalance/YYYY-MM-DD/`:

| File | Description |
|------|-------------|
| `picks.csv` | Target portfolio: ticker, weight, shares, price |
| `trades.csv` | Buy/sell orders needed |
| `portfolio_summary.json` | Regime, sector weights, parameters |
| `schedule.json` | Next 10 rebalance dates |
| `alpaca_orders_*.csv` | Order execution results |

---

## Troubleshooting

### "ModuleNotFoundError" or command not found
```bash
# Forgot to activate conda
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
```

### "Not a rebalance day" warning
```bash
# Use --force to generate anyway (for testing)
python scripts/production/generate_rebalance.py --db data/kairos.duckdb --date 2026-01-22 --force
```

### Data is stale
Run the pipeline to update features:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu && \
python scripts/run_pipeline.py --db data/kairos.duckdb --date 2026-01-23
```

### Orders go to "accepted" but not "filled"
- Normal if market is closed
- Orders execute at next market open (9:30 AM ET)
- Check Alpaca dashboard for order status

### "asset is not active" errors
- Stock is delisted or not tradable on Alpaca
- Safe to ignore - other orders still execute
- Will be excluded in next rebalance automatically

---

## Related Documentation

- Full pipeline guide: `docs/KAIROS_PIPELINE_GUIDE.md`
- Main project reference: `CLAUDE.md`
- Rebalance design spec: `rebalance-design.md`

---

*End of Quick Reference Guide*
