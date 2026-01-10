# Kairos Production Rebalance System Design

**Date:** December 31, 2025  
**Author:** Quant Engineering  
**Version:** 1.0

---

## Executive Summary

This document describes the production system for generating weekly stock picks and trade orders from the Kairos quantitative pipeline. The system determines when to rebalance, selects stocks using the Risk4 methodology, and generates actionable trade lists.

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WEEKLY REBALANCE WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   SCHEDULE  │───▶│    DATA     │───▶│   STOCK     │───▶│    TRADE    │  │
│  │   CHECK     │    │   CHECK     │    │  SELECTION  │    │ GENERATION  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│   Is today a         Is data           Apply Risk4       Compare to        │
│   rebalance day?     current?          methodology       current holdings  │
│                                                                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         OUTPUT ARTIFACTS                              │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │  • picks.csv          - Ranked stock list with target weights         │  │
│  │  • trades.csv         - Buy/sell orders                               │  │
│  │  • portfolio.json     - Summary metrics and regime indicator          │  │
│  │  • schedule.json      - Next 10 rebalance dates                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| REBALANCE_FREQUENCY | 5 | Trading days between rebalances |
| TOP_N | 75 | Number of stocks to hold |
| TARGET_VOL | 0.20 | Annual volatility target (20%) |
| MIN_ADV | 2,000,000 | Minimum average daily volume ($) |
| MAX_POSITION_PCT | 0.03 | Maximum single position (3%) |
| MAX_SECTOR_MULT | 2.0 | Max sector weight = 2× universe weight |
| LAMBDA_TC | 0.5 | Turnover smoothing (0=all new, 1=no change) |
| MAX_TURNOVER | 0.30 | Maximum portfolio turnover per rebalance |

---

## 2. Rebalance Schedule Logic

### 2.1 Trading Calendar

The system uses the NYSE trading calendar to determine valid trading days. Key considerations:

1. **Market Holidays** - NYSE closed (no rebalance)
2. **Half Days** - Treated as full trading days
3. **Weekend Gaps** - Skipped in day count

### 2.2 Rebalance Day Determination

```python
def is_rebalance_day(date, anchor_date, frequency=5):
    """
    Determine if a given date is a rebalance day.
    
    Algorithm:
    1. Get list of trading days from anchor to date
    2. Count trading days since anchor
    3. Return True if count % frequency == 0
    """
    trading_days = get_trading_days(anchor_date, date)
    days_since_anchor = len(trading_days) - 1
    return days_since_anchor % frequency == 0
```

### 2.3 Anchor Date Strategy

**Option A: Fixed Anchor (Recommended)**
- Use a fixed historical date (e.g., 2015-01-02, first trading day of backtest period)
- Ensures consistency with backtest results
- Simple to implement

**Option B: Rolling Anchor**
- Reset anchor on significant events (new year, strategy change)
- More flexible but can cause drift from backtest cadence

**Recommendation:** Use fixed anchor matching backtest start date to ensure production trades align with backtest assumptions.

### 2.4 Implementation

```python
# Configuration
ANCHOR_DATE = "2015-01-02"  # First trading day of backtest
FREQUENCY = 5               # Every 5 trading days

# Check if today is rebalance day
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def get_trading_calendar():
    """Get NYSE trading calendar."""
    holidays = USFederalHolidayCalendar()
    trading_day = CustomBusinessDay(calendar=holidays)
    return trading_day

def get_trading_days_since_anchor(target_date, anchor=ANCHOR_DATE):
    """Count trading days from anchor to target."""
    trading_day = get_trading_calendar()
    anchor_dt = pd.Timestamp(anchor)
    target_dt = pd.Timestamp(target_date)
    
    # Generate all trading days in range
    days = pd.date_range(start=anchor_dt, end=target_dt, freq=trading_day)
    return len(days) - 1  # Exclude anchor day

def is_rebalance_day(date):
    """Check if date is a rebalance day."""
    days = get_trading_days_since_anchor(date)
    return days % FREQUENCY == 0

def get_next_rebalance_date(from_date, n_future=10):
    """Get next N rebalance dates starting from given date."""
    trading_day = get_trading_calendar()
    current = pd.Timestamp(from_date)
    rebalance_dates = []
    
    while len(rebalance_dates) < n_future:
        if is_rebalance_day(current):
            rebalance_dates.append(current)
        current = current + trading_day
    
    return rebalance_dates
```

---

## 3. Stock Selection (Risk4 Methodology)

### 3.1 Selection Algorithm

The stock selection follows the Risk4 backtest methodology exactly:

```
Step 1: Load Data
    └── Pull latest date from feat_matrix_v2
    └── Join sector info from tickers table
    └── Require: alpha_composite_v8, vol_blend, adv_20, sector

Step 2: Filter Universe
    └── Drop rows with missing sector or vol_blend
    └── Apply ADV filter: adv_20 >= MIN_ADV

Step 3: Score and Rank
    └── Z-score alpha cross-sectionally
    └── Clip at ±3.0
    └── Rank by alpha_z descending
    └── Select top N stocks

Step 4: Calculate Weights
    └── Base weight = alpha_z / vol_blend (higher alpha, lower vol = more weight)
    └── Normalize to sum to 1.0
    └── Apply position cap: max 3% per stock
    └── Apply sector cap: max 2× universe weight
    └── Re-normalize

Step 5: Turnover Smoothing (if prior holdings exist)
    └── Blend: w_new = lambda * w_prior + (1-lambda) * w_target
    └── Cap turnover at MAX_TURNOVER
    └── Re-normalize

Step 6: Volatility Targeting
    └── Estimate portfolio vol from position vols
    └── Scale weights to achieve TARGET_VOL
```

### 3.2 SQL Query for Latest Data

```sql
SELECT 
    m.ticker,
    m.date,
    m.alpha_composite_v8,
    m.vol_blend,
    m.adv_20,
    t.sector
FROM feat_matrix_v2 m
JOIN tickers t ON m.ticker = t.ticker
WHERE m.date = (SELECT MAX(date) FROM feat_matrix_v2)
  AND m.alpha_composite_v8 IS NOT NULL
  AND m.vol_blend IS NOT NULL
  AND m.vol_blend > 0
  AND m.adv_20 >= 2000000
  AND t.sector IS NOT NULL
ORDER BY m.alpha_composite_v8 DESC;
```

### 3.3 Weight Calculation

```python
def calculate_weights(df, top_n=75, max_position=0.03, max_sector_mult=2.0):
    """
    Calculate portfolio weights using Risk4 methodology.
    
    Args:
        df: DataFrame with columns [ticker, alpha, vol_blend, sector]
        top_n: Number of stocks to select
        max_position: Maximum weight per stock
        max_sector_mult: Max sector weight as multiple of universe weight
    
    Returns:
        DataFrame with [ticker, weight, sector]
    """
    # Z-score alpha cross-sectionally
    df['alpha_z'] = (df['alpha'] - df['alpha'].mean()) / df['alpha'].std()
    df['alpha_z'] = df['alpha_z'].clip(-3, 3)
    
    # Select top N
    top = df.nlargest(top_n, 'alpha_z').copy()
    
    # Base weights: alpha / volatility
    top['raw_weight'] = top['alpha_z'] / top['vol_blend']
    top['raw_weight'] = top['raw_weight'].clip(lower=0)  # No negative weights
    top['weight'] = top['raw_weight'] / top['raw_weight'].sum()
    
    # Apply position cap
    top['weight'] = top['weight'].clip(upper=max_position)
    top['weight'] = top['weight'] / top['weight'].sum()
    
    # Apply sector cap
    universe_sector_weights = df.groupby('sector').size() / len(df)
    sector_caps = universe_sector_weights * max_sector_mult
    
    for sector in top['sector'].unique():
        mask = top['sector'] == sector
        sector_weight = top.loc[mask, 'weight'].sum()
        cap = sector_caps.get(sector, max_position)
        
        if sector_weight > cap:
            scale = cap / sector_weight
            top.loc[mask, 'weight'] *= scale
    
    # Re-normalize
    top['weight'] = top['weight'] / top['weight'].sum()
    
    return top[['ticker', 'weight', 'sector', 'alpha_z', 'vol_blend']]
```

---

## 4. Trade Generation

### 4.1 Holdings Comparison

```python
def generate_trades(target_weights, current_holdings, portfolio_value):
    """
    Generate trade list by comparing target to current holdings.
    
    Args:
        target_weights: DataFrame [ticker, weight]
        current_holdings: DataFrame [ticker, shares, current_price]
        portfolio_value: Total portfolio value in $
    
    Returns:
        DataFrame [ticker, action, shares, value, reason]
    """
    trades = []
    
    # Get all tickers (union of target and current)
    target_tickers = set(target_weights['ticker'])
    current_tickers = set(current_holdings['ticker'])
    all_tickers = target_tickers | current_tickers
    
    for ticker in all_tickers:
        target_weight = target_weights.loc[
            target_weights['ticker'] == ticker, 'weight'
        ].values
        target_weight = target_weight[0] if len(target_weight) > 0 else 0
        
        current_row = current_holdings[current_holdings['ticker'] == ticker]
        if len(current_row) > 0:
            current_shares = current_row['shares'].values[0]
            current_price = current_row['current_price'].values[0]
            current_value = current_shares * current_price
        else:
            current_shares = 0
            current_price = get_latest_price(ticker)  # Need to fetch
            current_value = 0
        
        target_value = target_weight * portfolio_value
        target_shares = int(target_value / current_price) if current_price > 0 else 0
        
        delta_shares = target_shares - current_shares
        delta_value = delta_shares * current_price
        
        if delta_shares != 0:
            trades.append({
                'ticker': ticker,
                'action': 'BUY' if delta_shares > 0 else 'SELL',
                'shares': abs(delta_shares),
                'value': abs(delta_value),
                'target_weight': target_weight,
                'current_shares': current_shares,
                'target_shares': target_shares,
                'price': current_price
            })
    
    return pd.DataFrame(trades)
```

### 4.2 Turnover Control

```python
def apply_turnover_control(target_weights, prior_weights, lambda_tc=0.5, max_turnover=0.30):
    """
    Blend target with prior weights to control turnover.
    
    Args:
        target_weights: DataFrame [ticker, weight]
        prior_weights: DataFrame [ticker, weight]
        lambda_tc: Smoothing parameter (0=all target, 1=all prior)
        max_turnover: Maximum allowed turnover
    
    Returns:
        DataFrame [ticker, weight]
    """
    # Merge on ticker
    merged = target_weights.merge(
        prior_weights, 
        on='ticker', 
        how='outer', 
        suffixes=('_target', '_prior')
    ).fillna(0)
    
    # Blend
    merged['weight_blended'] = (
        lambda_tc * merged['weight_prior'] + 
        (1 - lambda_tc) * merged['weight_target']
    )
    
    # Check turnover
    turnover = (merged['weight_blended'] - merged['weight_prior']).abs().sum() / 2
    
    if turnover > max_turnover:
        # Scale down the change
        scale = max_turnover / turnover
        merged['weight_blended'] = (
            merged['weight_prior'] + 
            scale * (merged['weight_blended'] - merged['weight_prior'])
        )
    
    # Normalize
    merged['weight'] = merged['weight_blended'] / merged['weight_blended'].sum()
    
    return merged[['ticker', 'weight']]
```

---

## 5. Output Artifacts

### 5.1 picks.csv

Stock picks with full detail for review:

```csv
ticker,weight,target_shares,sector,alpha_z,vol_blend,adv_20,rank
AAPL,0.0285,1425,Technology,2.34,0.18,125000000,1
MSFT,0.0274,890,Technology,2.21,0.16,98000000,2
...
```

### 5.2 trades.csv

Actionable trade list:

```csv
ticker,action,shares,estimated_value,price,reason
NVDA,BUY,50,6250.00,125.00,new_position
AAPL,BUY,25,4875.00,195.00,increase_weight
META,SELL,30,15000.00,500.00,decrease_weight
TSLA,SELL,100,24800.00,248.00,exit_position
```

### 5.3 portfolio_summary.json

```json
{
    "date": "2025-12-30",
    "is_rebalance_day": true,
    "next_rebalance": "2026-01-06",
    "regime": {
        "current": "normal_vol_bull",
        "vol_regime": "normal_vol",
        "trend_regime": "bull"
    },
    "portfolio": {
        "n_positions": 75,
        "total_weight": 1.0,
        "top_5_weight": 0.138,
        "sector_concentration": {
            "Technology": 0.28,
            "Healthcare": 0.15,
            "Financials": 0.12
        }
    },
    "metrics": {
        "estimated_beta": 1.02,
        "estimated_vol": 0.20,
        "avg_adv": 45000000,
        "turnover_vs_prior": 0.15
    },
    "data_freshness": {
        "feat_matrix_latest": "2025-12-30",
        "sep_base_latest": "2025-12-30",
        "regime_latest": "2025-12-30"
    }
}
```

### 5.4 schedule.json

```json
{
    "generated_at": "2025-12-30T08:00:00",
    "anchor_date": "2015-01-02",
    "frequency_days": 5,
    "next_10_rebalances": [
        "2026-01-06",
        "2026-01-13",
        "2026-01-21",
        "2026-01-27",
        "2026-02-03",
        "2026-02-10",
        "2026-02-18",
        "2026-02-24",
        "2026-03-03",
        "2026-03-10"
    ]
}
```

---

## 6. Integration with Weekly Pipeline

### 6.1 Extended Pipeline Flow

```
DAILY (data update):
    └── Download new market data
    └── Update sep_base
    └── Run Phase 1-5 pipeline
    └── Update feat_matrix_v2

WEEKLY (on rebalance day):
    └── All daily steps PLUS:
    └── Run generate_rebalance.py
    └── Generate picks.csv, trades.csv
    └── Generate portfolio_summary.json
    └── Email/notify for review
    └── Wait for manual approval
    └── Submit trades to broker
```

### 6.2 Pipeline Integration Points

Add to `run_pipeline.py`:

```python
# Phase 6: Production Rebalance (only on rebalance days)
if is_rebalance_day(current_date):
    logging.info("=" * 60)
    logging.info("PHASE 6: Production Rebalance")
    logging.info("=" * 60)
    
    run_script("python scripts/production/generate_rebalance.py "
               f"--db {db_path} "
               f"--date {current_date} "
               f"--output-dir outputs/rebalance/{current_date}")
```

---

## 7. Monitoring and Alerts

### 7.1 Pre-Rebalance Checks

Before generating picks, verify:

1. **Data Freshness**
   - feat_matrix_v2 has current date
   - All required columns present
   - No NULL anomalies in alpha column

2. **Signal Quality**
   - Alpha distribution is reasonable (not all same value)
   - Cross-sectional std > 0.1
   - Top-bottom spread > 1.0 std

3. **Universe Health**
   - >= 500 stocks pass ADV filter
   - All 11 GICS sectors represented
   - No single sector > 40% of universe

### 7.2 Alert Conditions

| Condition | Severity | Action |
|-----------|----------|--------|
| Data >1 day stale | CRITICAL | Block rebalance |
| <50 stocks pass filter | CRITICAL | Block rebalance |
| Turnover >50% | WARNING | Review before proceeding |
| Single position >5% | WARNING | Manual cap review |
| Regime change | INFO | Log for context |

---

## 8. Implementation Roadmap

### Phase 1: Core Script (Week 1)
- [ ] `generate_rebalance.py` - Main rebalance script
- [ ] Trading calendar integration
- [ ] Weight calculation (Risk4 methodology)
- [ ] Output file generation

### Phase 2: Integration (Week 2)
- [ ] Add Phase 6 to run_pipeline.py
- [ ] Prior holdings tracking (SQLite or DuckDB table)
- [ ] Turnover smoothing implementation

### Phase 3: Monitoring (Week 3)
- [ ] Pre-rebalance validation checks
- [ ] Alert system (email/Slack)
- [ ] Dashboard for portfolio state

### Phase 4: Execution (Week 4)
- [ ] Broker API integration (Interactive Brokers/Alpaca)
- [ ] Trade submission with approval workflow
- [ ] Execution reconciliation

---

## 9. Sample Commands

```bash
# Check if today is rebalance day
python scripts/production/check_rebalance.py --date 2025-12-30

# Generate rebalance (manual run)
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2025-12-30 \
    --output-dir outputs/rebalance/2025-12-30

# Generate rebalance with prior holdings
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb \
    --date 2025-12-30 \
    --prior-holdings outputs/rebalance/2025-12-23/picks.csv \
    --output-dir outputs/rebalance/2025-12-30

# View next 10 rebalance dates
python scripts/production/rebalance_schedule.py --next 10

# Validate data before rebalance
python scripts/production/validate_for_rebalance.py --db data/kairos.duckdb
```

---

## 10. Risk Controls Summary

| Control | Implementation | Purpose |
|---------|---------------|---------|
| ADV Filter | $2M minimum | Ensure liquidity |
| Position Cap | 3% maximum | Limit single-stock risk |
| Sector Cap | 2× universe weight | Diversification |
| Turnover Cap | 30% per rebalance | Trading cost control |
| Vol Targeting | 20% annual | Consistent risk exposure |
| Data Validation | Pre-rebalance checks | Data quality |

---

*End of Production Rebalance System Design*