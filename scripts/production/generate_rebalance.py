#!/usr/bin/env python3
"""
Kairos Production Rebalance Generator

Generates stock picks and trade lists for weekly rebalancing.
Implements Risk4 methodology for portfolio construction.

Usage:
    python generate_rebalance.py --db data/kairos.duckdb --date 2025-12-30

Author: Kairos Quant Engineering
Version: 1.1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Try to import pandas_market_calendars for NYSE calendar
try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CAL = True
except ImportError:
    HAS_MARKET_CAL = False
    logging.warning("pandas_market_calendars not installed. Using basic calendar.")

# ============================================================================
# CONFIGURATION
# ============================================================================

V8_CONFIG = {
    # Portfolio construction
    "top_n": 75,                  # Number of stocks to hold
    "target_vol": 0.20,           # 20% annual vol target
    "min_adv": 2_000_000,         # $2M minimum ADV
    "max_position_pct": 0.03,     # 3% max per position
    "max_sector_mult": 2.0,       # Max sector = 2x universe weight
    
    # Turnover control
    "lambda_tc": 0.5,             # Turnover smoothing (0=all new, 1=no change)
    "max_turnover": 0.30,         # 30% max turnover per rebalance
    
    # Alpha signal
    "alpha_column": "alpha_composite_v8",
    "vol_column": "vol_blend",
    "adv_column": "adv_20",
}

ML_CONFIG = {
    # Portfolio construction
    "top_n": 75,                  # Keep at 75 (best)
    "target_vol": 0.25,           # CHANGED: 25% (was 20%)
    "min_adv": 2_000_000,         # $2M minimum ADV
    "max_position_pct": 0.03,     # 3% max per position
    "max_sector_mult": 2.0,       # Max sector = 2x universe weight
    
    # Turnover control
    "lambda_tc": 0.5,             # Turnover smoothing (0=all new, 1=no change)
    "max_turnover": 0.30,         # 30% max turnover per rebalance
    
    # Alpha signal
    "alpha_column": "alpha_ml_v2_clf",  # CHANGED: ML signal (was v8)
    "vol_column": "vol_blend",
    "adv_column": "adv_20",
}

# Select configuration
CONFIG = ML_CONFIG.copy()

# Known NYSE holidays for 2024-2026 (fallback if no market calendar)
NYSE_HOLIDAYS = {
    # 2024
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', 
    '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
    '2024-11-28', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
    '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
    '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
    '2026-11-26', '2026-12-25',
}

# ============================================================================
# TRADING CALENDAR - Last Trading Day of Week
# ============================================================================

def is_trading_day(date):
    """Check if a date is a trading day (not weekend, not holiday)."""
    d = pd.Timestamp(date)
    
    # Weekend check
    if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Holiday check
    if HAS_MARKET_CAL:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=d, end_date=d)
        return len(schedule) > 0
    else:
        # Use hardcoded holidays
        return d.strftime('%Y-%m-%d') not in NYSE_HOLIDAYS


def get_trading_days_in_week(date):
    """Get all trading days in the same week as the given date."""
    d = pd.Timestamp(date)
    
    # Find Monday of this week
    monday = d - timedelta(days=d.weekday())
    
    # Get Mon-Fri of this week
    week_days = [monday + timedelta(days=i) for i in range(5)]
    
    # Filter to trading days only
    trading_days = [day for day in week_days if is_trading_day(day)]
    
    return trading_days


def get_last_trading_day_of_week(date):
    """Get the last trading day of the week containing the given date."""
    trading_days = get_trading_days_in_week(date)
    if trading_days:
        return trading_days[-1]
    return None


def is_rebalance_day(date):
    """Check if date is a rebalance day (last trading day of week)."""
    d = pd.Timestamp(date)
    
    if not is_trading_day(d):
        return False
    
    last_day = get_last_trading_day_of_week(d)
    return last_day is not None and d.date() == last_day.date()


def get_next_rebalance_dates(from_date, n_future=10):
    """Get next N rebalance dates (last trading day of each week)."""
    current = pd.Timestamp(from_date)
    rebalance_dates = []
    
    # Move to start of current week
    monday = current - timedelta(days=current.weekday())
    
    weeks_checked = 0
    while len(rebalance_dates) < n_future and weeks_checked < 100:
        week_start = monday + timedelta(weeks=weeks_checked)
        last_day = get_last_trading_day_of_week(week_start)
        
        if last_day is not None and last_day >= current:
            rebalance_dates.append(last_day)
        
        weeks_checked += 1
    
    return rebalance_dates

# ============================================================================
# DATA LOADING
# ============================================================================

def load_latest_data(con, target_date=None):
    """
    Load latest feature data for stock selection.
    
    Returns DataFrame with columns:
        ticker, date, alpha, vol_blend, adv_20, sector, price
    """
    alpha_col = CONFIG["alpha_column"]
    vol_col = CONFIG["vol_column"]
    adv_col = CONFIG["adv_column"]
    
    if target_date:
        date_clause = f"m.date = '{target_date}'"
    else:
        date_clause = "m.date = (SELECT MAX(date) FROM feat_matrix_v2)"
    
    query = f"""
    SELECT 
        m.ticker,
        m.date,
        m.{alpha_col} as alpha,
        m.{vol_col} as vol_blend,
        m.{adv_col} as adv_20,
        t.sector,
        s.close as price
    FROM feat_matrix_v2 m
    JOIN tickers t ON m.ticker = t.ticker
    JOIN sep_base_academic s ON m.ticker = s.ticker AND m.date = s.date
    WHERE {date_clause}
      AND m.{alpha_col} IS NOT NULL
      AND m.{vol_col} IS NOT NULL
      AND m.{vol_col} > 0
      AND m.{adv_col} >= {CONFIG['min_adv']}
      AND t.sector IS NOT NULL
      AND t."table" = 'SEP'
    ORDER BY m.{alpha_col} DESC
    """
    
    df = con.execute(query).fetchdf()
    logging.info(f"Loaded {len(df)} stocks passing filters")
    
    return df

def load_regime(con, target_date=None):
    """Load current market regime."""
    if target_date:
        date_clause = f"date = '{target_date}'"
    else:
        date_clause = "date = (SELECT MAX(date) FROM regime_history_academic)"
    
    query = f"""
    SELECT date, vol_regime, trend_regime, regime
    FROM regime_history_academic
    WHERE {date_clause}
    LIMIT 1
    """
    
    result = con.execute(query).fetchdf()
    if len(result) == 0:
        return {"regime": "unknown", "vol_regime": "unknown", "trend_regime": "unknown"}
    
    row = result.iloc[0]
    return {
        "date": str(row['date']),
        "regime": row['regime'],
        "vol_regime": row['vol_regime'],
        "trend_regime": row['trend_regime']
    }

def load_data_freshness(con):
    """Check data freshness across key tables."""
    tables = [
        ("feat_matrix_v2", "date"),
        ("sep_base_academic", "date"),
        ("regime_history_academic", "date"),
        ("feat_composite_v8", "date"),
    ]
    
    freshness = {}
    for table, date_col in tables:
        try:
            result = con.execute(f"SELECT MAX({date_col}) as max_date FROM {table}").fetchone()
            freshness[table] = str(result[0]) if result[0] else "N/A"
        except:
            freshness[table] = "ERROR"
    
    return freshness

# ============================================================================
# WEIGHT CALCULATION (Risk4 Methodology)
# ============================================================================

def calculate_weights(df, prior_weights=None):
    """
    Calculate portfolio weights using Risk4 methodology.
    
    Steps:
        1. Z-score alpha cross-sectionally
        2. Clip outliers at ±3
        3. Select top N
        4. Weight by alpha_z / vol_blend
        5. Apply position caps
        6. Apply sector caps
        7. Apply turnover smoothing (if prior weights provided)
        8. Normalize
    """
    top_n = CONFIG["top_n"]
    max_pos = CONFIG["max_position_pct"]
    max_sector = CONFIG["max_sector_mult"]
    
    # Step 1 & 2: Z-score and clip
    df = df.copy()
    df['alpha_z'] = (df['alpha'] - df['alpha'].mean()) / df['alpha'].std()
    df['alpha_z'] = df['alpha_z'].clip(-3, 3)
    
    # Step 3: Select top N
    top = df.nlargest(top_n, 'alpha_z').copy()
    top['rank'] = range(1, len(top) + 1)
    
    # Step 4: Base weights = alpha_z / vol_blend
    # Higher alpha and lower vol = higher weight
    top['raw_weight'] = top['alpha_z'] / top['vol_blend']
    top['raw_weight'] = top['raw_weight'].clip(lower=0.001)  # Ensure positive
    top['weight'] = top['raw_weight'] / top['raw_weight'].sum()
    
    # Step 5: Position cap
    top['weight'] = top['weight'].clip(upper=max_pos)
    top['weight'] = top['weight'] / top['weight'].sum()
    
    # Step 6: Sector cap
    # Calculate universe sector weights
    universe_sector_counts = df.groupby('sector').size()
    universe_sector_weights = universe_sector_counts / len(df)
    
    for sector in top['sector'].unique():
        mask = top['sector'] == sector
        sector_weight = top.loc[mask, 'weight'].sum()
        cap = universe_sector_weights.get(sector, max_pos) * max_sector
        cap = min(cap, 0.40)  # Hard cap at 40% regardless
        
        if sector_weight > cap:
            scale = cap / sector_weight
            top.loc[mask, 'weight'] *= scale
            logging.info(f"Capped {sector} from {sector_weight:.1%} to {cap:.1%}")
    
    # Re-normalize after sector caps
    top['weight'] = top['weight'] / top['weight'].sum()
    
    # Step 7: Turnover smoothing (if prior weights provided)
    if prior_weights is not None:
        top = apply_turnover_smoothing(top, prior_weights)
    
    # Final normalization
    top['weight'] = top['weight'] / top['weight'].sum()
    
    return top

def apply_turnover_smoothing(target, prior_weights):
    """Blend target weights with prior weights to control turnover."""
    lambda_tc = CONFIG["lambda_tc"]
    max_turnover = CONFIG["max_turnover"]
    
    # Merge with prior
    prior_df = prior_weights[['ticker', 'weight']].copy()
    prior_df.columns = ['ticker', 'weight_prior']
    
    merged = target.merge(prior_df, on='ticker', how='left')
    merged['weight_prior'] = merged['weight_prior'].fillna(0)
    
    # Blend
    merged['weight_blended'] = (
        lambda_tc * merged['weight_prior'] + 
        (1 - lambda_tc) * merged['weight']
    )
    
    # Check turnover
    turnover = (merged['weight_blended'] - merged['weight_prior']).abs().sum() / 2
    logging.info(f"Turnover before cap: {turnover:.1%}")
    
    if turnover > max_turnover:
        scale = max_turnover / turnover
        merged['weight_blended'] = (
            merged['weight_prior'] + 
            scale * (merged['weight_blended'] - merged['weight_prior'])
        )
        logging.info(f"Turnover capped to: {max_turnover:.1%}")
    
    merged['weight'] = merged['weight_blended']
    
    return merged.drop(columns=['weight_prior', 'weight_blended'])

# ============================================================================
# TRADE GENERATION
# ============================================================================

def generate_trades(target_weights, current_holdings=None, portfolio_value=1_000_000):
    """
    Generate trade list comparing target to current holdings.
    """
    if current_holdings is None:
        # First rebalance - all buys
        trades = target_weights[['ticker', 'weight']].copy()
        if 'price' in target_weights.columns:
            trades['price'] = target_weights['price']
        trades['action'] = 'BUY'
        trades['target_value'] = (trades['weight'] * portfolio_value).round(2)
        trades['current_value'] = 0
        trades['delta_value'] = trades['target_value']
        return trades
    
    # Rename prior weight column to avoid merge conflict
    prior = current_holdings[['ticker', 'weight']].copy()
    prior.columns = ['ticker', 'prior_weight']
    
    # Merge target with prior
    merged = target_weights[['ticker', 'weight']].merge(
        prior,
        on='ticker',
        how='outer'
    ).fillna(0)
    
    # Add price from target_weights
    if 'price' in target_weights.columns:
        price_df = target_weights[['ticker', 'price']].copy()
        merged = merged.merge(price_df, on='ticker', how='left')
    
    merged['target_value'] = merged['weight'] * portfolio_value
    merged['current_value'] = merged['prior_weight'] * portfolio_value
    merged['delta_value'] = merged['target_value'] - merged['current_value']
    
    # Determine action
    merged['action'] = 'HOLD'
    merged.loc[merged['delta_value'] > 100, 'action'] = 'BUY'
    merged.loc[merged['delta_value'] < -100, 'action'] = 'SELL'
    
    # Filter to actual trades
    trades = merged[merged['action'] != 'HOLD'].copy()
    trades['delta_value'] = trades['delta_value'].abs()
    
    return trades.sort_values('delta_value', ascending=False)

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_picks_csv(weights, output_path, portfolio_value=1_000_000):
    """Generate picks.csv with full detail including price and shares."""
    cols = ['rank', 'ticker', 'weight', 'sector', 'alpha_z', 'vol_blend', 'adv_20', 'price']
    available_cols = [c for c in cols if c in weights.columns]
    
    output = weights[available_cols].copy()
    output['weight'] = output['weight'].round(6)
    output['alpha_z'] = output['alpha_z'].round(4) if 'alpha_z' in output.columns else None
    
    # Add target value and shares
    if 'price' in output.columns:
        output['price'] = output['price'].round(2)
        output['target_value'] = (output['weight'] * portfolio_value).round(2)
        output['shares'] = (output['target_value'] / output['price']).astype(int)
    
    output.to_csv(output_path, index=False)
    logging.info(f"Wrote {len(output)} picks to {output_path}")

def generate_trades_csv(trades, target_weights, output_path):
    """Generate trades.csv with actionable orders."""
    if len(trades) == 0:
        logging.info("No trades to generate")
        return
    
    # Build output dataframe
    output = trades[['ticker', 'action', 'delta_value', 'target_value', 'current_value']].copy()
    output.columns = ['ticker', 'action', 'trade_value', 'target_value', 'current_value']
    output['trade_value'] = output['trade_value'].round(2)
    output['target_value'] = output['target_value'].round(2)
    output['current_value'] = output['current_value'].round(2)
    
    # Add price and shares if available
    if 'price' in trades.columns:
        output['price'] = trades['price'].fillna(0).round(2)
        output['shares'] = output.apply(
            lambda row: int(row['trade_value'] / row['price']) if row['price'] > 0 else 0, 
            axis=1
        )
    
    output.to_csv(output_path, index=False)
    logging.info(f"Wrote {len(output)} trades to {output_path}")

def generate_portfolio_summary(weights, regime, freshness, target_date, output_path):
    """Generate portfolio_summary.json."""
    # Calculate metrics
    n_positions = len(weights)
    top_5_weight = weights.nlargest(5, 'weight')['weight'].sum()
    
    sector_weights = weights.groupby('sector')['weight'].sum().to_dict()
    
    # Get next rebalance
    next_rebalances = get_next_rebalance_dates(target_date, n_future=1)
    next_rebalance = str(next_rebalances[0].date()) if next_rebalances else "unknown"
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "rebalance_date": str(target_date),
        "is_rebalance_day": is_rebalance_day(target_date),
        "next_rebalance": next_rebalance,
        "regime": regime,
        "portfolio": {
            "n_positions": n_positions,
            "top_n_target": CONFIG["top_n"],
            "total_weight": round(weights['weight'].sum(), 6),
            "top_5_weight": round(top_5_weight, 4),
            "sector_weights": {k: round(v, 4) for k, v in sector_weights.items()}
        },
        "parameters": {
            "alpha_column": CONFIG["alpha_column"],
            "target_vol": CONFIG["target_vol"],
            "min_adv": CONFIG["min_adv"],
            "max_position_pct": CONFIG["max_position_pct"],
            "rebalance_rule": "Last trading day of each week"
        },
        "data_freshness": freshness
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Wrote portfolio summary to {output_path}")

def generate_schedule_json(from_date, output_path, n_future=10):
    """Generate schedule.json with upcoming rebalance dates."""
    next_dates = get_next_rebalance_dates(from_date, n_future)
    
    schedule = {
        "generated_at": datetime.now().isoformat(),
        "rule": "Last trading day of each week",
        "next_rebalances": [str(d.date()) for d in next_dates]
    }
    
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    
    logging.info(f"Wrote schedule to {output_path}")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_data(df, target_date):
    """Run pre-rebalance validation checks."""
    issues = []
    warnings = []
    
    # Check 1: Minimum stocks
    if len(df) < 100:
        issues.append(f"CRITICAL: Only {len(df)} stocks pass filters (need >= 100)")
    elif len(df) < 200:
        warnings.append(f"WARNING: Only {len(df)} stocks pass filters")
    
    # Check 2: Alpha distribution
    alpha_std = df['alpha'].std()
    if alpha_std < 0.01:
        issues.append(f"CRITICAL: Alpha std = {alpha_std:.4f} (near zero variance)")
    
    # Check 3: Sector representation
    sectors = df['sector'].nunique()
    if sectors < 8:
        warnings.append(f"WARNING: Only {sectors} sectors represented")
    
    # Check 4: Date match
    data_date = df['date'].iloc[0] if len(df) > 0 else None
    if data_date and str(data_date) != str(target_date):
        warnings.append(f"WARNING: Data date {data_date} != target {target_date}")
    
    return issues, warnings

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Kairos rebalance picks')
    parser.add_argument('--db', required=True, help='Path to DuckDB database')
    parser.add_argument('--date', help='Target date (default: latest in database)')
    parser.add_argument('--output-dir', default='outputs/rebalance', help='Output directory')
    parser.add_argument('--prior-holdings', help='Path to prior picks.csv for turnover smoothing')
    parser.add_argument('--portfolio-value', type=float, default=100_000, help='Portfolio value in $')
    parser.add_argument('--check-only', action='store_true', help='Only check if rebalance day')
    parser.add_argument('--force', action='store_true', help='Generate even if not rebalance day')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Connect to database
    con = duckdb.connect(args.db, read_only=True)
    
    # Determine target date
    if args.date:
        target_date = args.date
    else:
        result = con.execute("SELECT MAX(date) FROM feat_matrix_v2").fetchone()
        target_date = str(result[0])
    
    logging.info(f"Target date: {target_date}")
    
    # Check if rebalance day
    is_rebal = is_rebalance_day(target_date)
    logging.info(f"Is rebalance day: {is_rebal}")
    
    if args.check_only:
        print(f"Date: {target_date}")
        print(f"Is rebalance day: {is_rebal}")
        next_dates = get_next_rebalance_dates(target_date, 5)
        print(f"Next 5 rebalance dates: {[str(d.date()) for d in next_dates]}")
        return 0
    
    if not is_rebal and not args.force:
        logging.warning("Not a rebalance day. Use --force to generate anyway.")
        return 0
    
    # Create output directory
    output_dir = Path(args.output_dir) / target_date
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Load data
    logging.info("Loading latest data...")
    df = load_latest_data(con, target_date)
    regime = load_regime(con, target_date)
    freshness = load_data_freshness(con)
    
    # Validate
    issues, warnings = validate_data(df, target_date)
    for w in warnings:
        logging.warning(w)
    for i in issues:
        logging.error(i)
    
    if issues:
        logging.error("Validation failed. Aborting.")
        return 1
    
    # Load prior weights if provided
    prior_weights = None
    if args.prior_holdings:
        try:
            prior_weights = pd.read_csv(args.prior_holdings)
            logging.info(f"Loaded {len(prior_weights)} prior holdings")
        except Exception as e:
            logging.warning(f"Could not load prior holdings: {e}")
    
    # Calculate weights
    logging.info("Calculating portfolio weights...")
    weights = calculate_weights(df, prior_weights)
    
    # Generate trades
    logging.info("Generating trade list...")
    trades = generate_trades(weights, prior_weights, args.portfolio_value)
    
    # Generate outputs
    logging.info("Generating output files...")
    generate_picks_csv(weights, output_dir / "picks.csv", args.portfolio_value)
    generate_trades_csv(trades, weights, output_dir / "trades.csv")
    generate_portfolio_summary(weights, regime, freshness, target_date, output_dir / "portfolio_summary.json")
    generate_schedule_json(target_date, output_dir / "schedule.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"REBALANCE GENERATED: {target_date}")
    print("=" * 60)
    print(f"Regime: {regime.get('regime', 'unknown')}")
    print(f"Positions: {len(weights)}")
    print(f"Top 5 weight: {weights.nlargest(5, 'weight')['weight'].sum():.1%}")
    print(f"\nTop 10 picks:")
    print(weights[['ticker', 'weight', 'sector']].head(10).to_string(index=False))
    print(f"\nOutputs in: {output_dir}")
    print("=" * 60)
    
    con.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())