#!/usr/bin/env python3
"""
analyze_turnover_costs.py
=========================
Diagnostic script to analyze realized turnover and estimate transaction costs
for the Kairos Risk4 long-only strategy.

This script:
1. Re-runs the Risk4 backtest logic while tracking detailed turnover metrics
2. Records per-rebalance: turnover (one-way), number of trades, avg trade size
3. Applies multiple transaction cost models to estimate net returns
4. Calculates gross vs net Sharpe ratios

Transaction Cost Models:
- Model A (Retail): 0.10% round-trip (commission + half-spread for liquid stocks)
- Model B (Institutional): 0.20% round-trip (includes some market impact)
- Model C (Conservative): 0.40% round-trip (significant market impact)

Usage:
    python analyze_turnover_costs.py --db data/kairos.duckdb

Output:
    - Console summary of turnover statistics
    - Gross vs Net performance under each cost model
    - Breakeven cost analysis
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - matches Risk4 defaults
# =============================================================================
DEFAULT_CONFIG = {
    "alpha_column": "alpha_composite_v8",
    "target_column": "ret_5d_f",
    "top_n": 75,
    "rebalance_every": 5,
    "target_vol": 0.20,
    "adv_thresh": 2_000_000.0,
    "sector_cap_mult": 2.0,
    "max_stock_w": 0.03,
    "lambda_tc": 0.20,
    "turnover_cap": 0.20,
    "start_date": "2015-01-01",
    "end_date": "2025-12-07",
}

# Transaction cost models (round-trip costs as fraction)
COST_MODELS = {
    "A_Retail": 0.0010,        # 0.10% = 5bps each way
    "B_Institutional": 0.0020, # 0.20% = 10bps each way  
    "C_Conservative": 0.0040,  # 0.40% = 20bps each way
    "D_HighImpact": 0.0060,    # 0.60% = 30bps each way
}


# =============================================================================
# Utility Functions
# =============================================================================

def safe_z(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score with safe fallback."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq: pd.Series) -> float:
    """Max drawdown on equity curve."""
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> Tuple[float, float, float]:
    """Annualized return, vol, Sharpe for periodic returns."""
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(con: duckdb.DuckDBPyConnection, config: dict) -> pd.DataFrame:
    """Load required data from feat_matrix + tickers."""
    logger.info("Loading data from feat_matrix...")
    
    alpha_col = config["alpha_column"]
    target_col = config["target_column"]
    start = config["start_date"]
    end = config["end_date"]
    
    # Check which table exists
    tables = con.execute("SHOW TABLES").fetchdf()
    table_names = tables['name'].tolist()
    
    if 'feat_matrix_v2' in table_names:
        matrix_table = 'feat_matrix_v2'
    elif 'feat_matrix' in table_names:
        matrix_table = 'feat_matrix'
    else:
        raise ValueError("No feat_matrix_v2 or feat_matrix table found")
    
    logger.info(f"Using table: {matrix_table}")
    
    # Check available columns
    cols = con.execute(f"DESCRIBE {matrix_table}").fetchdf()
    available_cols = cols['column_name'].tolist()
    
    if alpha_col not in available_cols:
        logger.warning(f"Column {alpha_col} not found. Available alpha columns:")
        alpha_cols = [c for c in available_cols if 'alpha' in c.lower()]
        for c in alpha_cols[:10]:
            logger.warning(f"  - {c}")
        raise ValueError(f"Alpha column {alpha_col} not in table")
    
    where = [
        f"{alpha_col} IS NOT NULL",
        f"{target_col} IS NOT NULL",
        "vol_blend IS NOT NULL",
        "adv_20 IS NOT NULL",
    ]
    if start:
        where.append(f"date >= DATE '{start}'")
    if end:
        where.append(f"date <= DATE '{end}'")
    
    where_sql = " AND ".join(where)
    
    query = f"""
        SELECT
            fm.ticker,
            fm.date,
            fm.{alpha_col} AS alpha,
            fm.{target_col} AS target,
            fm.vol_blend,
            fm.adv_20,
            t.sector
        FROM {matrix_table} fm
        LEFT JOIN tickers t USING(ticker)
        WHERE {where_sql}
        ORDER BY date, ticker
    """
    
    df = con.execute(query).fetchdf()
    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates")
    
    return df


# =============================================================================
# Core Backtest with Turnover Tracking
# =============================================================================

def run_backtest_with_turnover(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Risk4 backtest while tracking detailed turnover metrics.
    
    Returns:
        bt_df: DataFrame with returns per period
        turnover_df: DataFrame with turnover details per rebalance
    """
    top_n = config["top_n"]
    rebalance_every = config["rebalance_every"]
    adv_thresh = config["adv_thresh"]
    sector_cap_mult = config["sector_cap_mult"]
    max_stock_w = config["max_stock_w"]
    lambda_tc = config["lambda_tc"]
    turnover_cap = config["turnover_cap"]
    
    # Clean data
    logger.info("Cleaning data...")
    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    
    if df.empty:
        logger.error("No data after filtering")
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute alpha z-scores
    logger.info("Computing alpha z-scores...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z).clip(-3, 3)
    
    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]
    
    logger.info(f"Running backtest over {len(rebal_dates)} rebalance periods...")
    
    w_old = pd.Series(dtype=float)
    return_records = []
    turnover_records = []
    
    for i, d in enumerate(rebal_dates):
        day = df[df["date"] == d].copy()
        if day.empty:
            continue
        
        universe = day.copy()
        
        # Select top N by alpha
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        picks = picks.drop_duplicates(subset="ticker").set_index("ticker")
        
        if picks.empty:
            continue
        
        # Initial weights: alpha_z / vol_blend
        w_prop = picks["alpha_z"].clip(lower=0.0)
        if w_prop.sum() == 0:
            continue
        
        w_prop = w_prop / picks["vol_blend"].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / w_prop.sum()
        
        # Cap per-stock weight
        w_prop = w_prop.clip(upper=max_stock_w)
        w_prop = w_prop / w_prop.sum()
        
        # Sector caps
        sector_counts = universe.groupby("sector")["ticker"].count()
        total_univ = float(len(universe))
        sector_univ_w = sector_counts / total_univ
        
        for sec in picks["sector"].unique():
            sec_mask = picks["sector"] == sec
            w_sec = w_prop[sec_mask].sum()
            cap = sector_univ_w.get(sec, 0.0) * sector_cap_mult
            if cap > 0 and w_sec > cap:
                scale = cap / w_sec
                w_prop.loc[sec_mask] *= scale
        
        if w_prop.sum() <= 0:
            continue
        w_prop = w_prop / w_prop.sum()
        
        # Turnover control
        if w_old.empty:
            w_new = w_prop.copy()
            turnover_oneway = w_prop.sum()  # 100% turnover on first period
            turnover_raw = turnover_oneway
            turnover_capped = False
        else:
            all_tickers = w_old.index.union(w_prop.index).unique()
            w_old_full = w_old.reindex(all_tickers).fillna(0.0)
            w_prop_full = w_prop.reindex(all_tickers).fillna(0.0)
            
            # Blend old and new
            w_pre = (1 - lambda_tc) * w_old_full + lambda_tc * w_prop_full
            
            # Calculate raw turnover (before capping)
            turnover_raw = float((w_pre - w_old_full).abs().sum())
            
            # Apply turnover cap
            if turnover_raw > turnover_cap:
                scale = turnover_cap / turnover_raw
                w_new_full = w_old_full + scale * (w_pre - w_old_full)
                turnover_capped = True
            else:
                w_new_full = w_pre
                turnover_capped = False
            
            # Normalize
            if w_new_full.sum() <= 0:
                w_new_full = w_old_full
            w_new_full = w_new_full / w_new_full.sum()
            
            # Restrict to current picks
            w_new = w_new_full.reindex(w_prop.index).fillna(0.0)
            
            # Calculate actual turnover after capping
            w_old_aligned = w_old.reindex(w_new.index).fillna(0.0)
            turnover_oneway = float((w_new - w_old_aligned).abs().sum()) / 2
        
        # Count trades
        if w_old.empty:
            n_buys = len(w_new[w_new > 0])
            n_sells = 0
            n_holds = 0
        else:
            w_old_aligned = w_old.reindex(w_new.index).fillna(0.0)
            buys = (w_new > w_old_aligned) & (w_new > 0)
            sells = (w_new < w_old_aligned) | ((w_old > 0) & (~w_old.index.isin(w_new.index)))
            n_buys = buys.sum()
            n_sells = len(w_old) - len(w_old.index.intersection(w_new.index))
            n_sells += (w_new < w_old_aligned).sum()
            n_holds = len(w_new) - n_buys
        
        # Record turnover details
        turnover_records.append({
            "date": d,
            "period": i,
            "n_positions": len(w_new[w_new > 0]),
            "turnover_raw": turnover_raw,
            "turnover_oneway": turnover_oneway,
            "turnover_capped": turnover_capped,
            "n_buys": int(n_buys),
            "n_sells": int(n_sells),
            "avg_position_size": float(w_new.mean()),
            "max_position_size": float(w_new.max()),
        })
        
        w_old = w_new.copy()
        
        # Compute returns
        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue
        
        port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        bench_ret = float(universe["target"].mean())
        
        return_records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
            "turnover_oneway": turnover_oneway,
        })
    
    bt_df = pd.DataFrame(return_records).sort_values("date").reset_index(drop=True)
    turnover_df = pd.DataFrame(turnover_records).sort_values("date").reset_index(drop=True)
    
    return bt_df, turnover_df


# =============================================================================
# Cost Analysis
# =============================================================================

def apply_transaction_costs(
    bt_df: pd.DataFrame,
    roundtrip_cost: float,
    rebalance_every: int
) -> pd.DataFrame:
    """
    Apply transaction costs to returns.
    
    Args:
        bt_df: DataFrame with port_ret_raw and turnover_oneway
        roundtrip_cost: Cost as fraction (e.g., 0.002 = 20bps)
        rebalance_every: Days between rebalances
        
    Returns:
        DataFrame with net returns added
    """
    df = bt_df.copy()
    
    # Cost per period = turnover × roundtrip_cost
    # turnover_oneway is already the fraction of portfolio traded
    # For roundtrip, we pay on both buy and sell, so:
    # cost = turnover_oneway × roundtrip_cost
    # (turnover_oneway already accounts for one side, multiply by 2 is implicit in roundtrip)
    
    # Actually, turnover_oneway is sum(|w_new - w_old|)/2, which is the one-way turnover
    # Roundtrip cost applies to the full trade, so:
    # cost = turnover_oneway × roundtrip_cost
    
    df["cost"] = df["turnover_oneway"] * roundtrip_cost
    df["port_ret_net"] = df["port_ret_raw"] - df["cost"]
    
    return df


def calculate_net_performance(
    bt_df: pd.DataFrame,
    cost_models: Dict[str, float],
    config: dict
) -> pd.DataFrame:
    """Calculate net performance under each cost model."""
    
    rebalance_every = config["rebalance_every"]
    target_vol = config["target_vol"]
    
    results = []
    
    # First, calculate gross stats
    raw_returns = bt_df["port_ret_raw"]
    ann_ret_raw, ann_vol_raw, _ = annualize(raw_returns, rebalance_every)
    
    # Vol scaling factor
    scale = target_vol / ann_vol_raw if ann_vol_raw > 0 else 1.0
    
    # Gross performance (vol-targeted)
    gross_returns = raw_returns * scale
    gross_ann_ret, gross_ann_vol, gross_sharpe = annualize(gross_returns, rebalance_every)
    gross_equity = (1 + gross_returns).cumprod()
    gross_dd = max_drawdown(gross_equity)
    
    results.append({
        "Model": "GROSS (no costs)",
        "Cost_bps": 0,
        "Ann_Return": gross_ann_ret,
        "Ann_Vol": gross_ann_vol,
        "Sharpe": gross_sharpe,
        "Max_DD": gross_dd,
        "Total_Return": float(gross_equity.iloc[-1] - 1),
    })
    
    # Net performance under each cost model
    for model_name, roundtrip_cost in cost_models.items():
        bt_net = apply_transaction_costs(bt_df, roundtrip_cost, rebalance_every)
        
        # Net returns, vol-targeted
        net_returns = bt_net["port_ret_net"] * scale
        net_ann_ret, net_ann_vol, net_sharpe = annualize(net_returns, rebalance_every)
        net_equity = (1 + net_returns).cumprod()
        net_dd = max_drawdown(net_equity)
        
        results.append({
            "Model": model_name,
            "Cost_bps": roundtrip_cost * 10000,
            "Ann_Return": net_ann_ret,
            "Ann_Vol": net_ann_vol,
            "Sharpe": net_sharpe,
            "Max_DD": net_dd,
            "Total_Return": float(net_equity.iloc[-1] - 1),
        })
    
    return pd.DataFrame(results)


def calculate_breakeven_cost(bt_df: pd.DataFrame, config: dict) -> float:
    """
    Calculate the breakeven transaction cost (where Sharpe = 0).
    Uses binary search.
    """
    rebalance_every = config["rebalance_every"]
    target_vol = config["target_vol"]
    
    raw_returns = bt_df["port_ret_raw"]
    ann_ret_raw, ann_vol_raw, _ = annualize(raw_returns, rebalance_every)
    scale = target_vol / ann_vol_raw if ann_vol_raw > 0 else 1.0
    
    # Binary search for breakeven cost
    low, high = 0.0, 0.10  # 0% to 10% roundtrip cost
    
    for _ in range(50):
        mid = (low + high) / 2
        bt_net = apply_transaction_costs(bt_df, mid, rebalance_every)
        net_returns = bt_net["port_ret_net"] * scale
        _, _, sharpe = annualize(net_returns, rebalance_every)
        
        if sharpe > 0:
            low = mid
        else:
            high = mid
    
    return mid


# =============================================================================
# Reporting
# =============================================================================

def print_turnover_summary(turnover_df: pd.DataFrame) -> None:
    """Print summary statistics for turnover."""
    
    # Exclude first period (100% turnover on initialization)
    t = turnover_df.iloc[1:].copy() if len(turnover_df) > 1 else turnover_df
    
    print("\n" + "=" * 70)
    print("TURNOVER ANALYSIS")
    print("=" * 70)
    
    print(f"\nRebalance Periods: {len(turnover_df)}")
    print(f"Date Range: {turnover_df['date'].min()} to {turnover_df['date'].max()}")
    
    print("\n--- One-Way Turnover per Rebalance (excluding first period) ---")
    print(f"  Mean:    {t['turnover_oneway'].mean():.2%}")
    print(f"  Median:  {t['turnover_oneway'].median():.2%}")
    print(f"  Std:     {t['turnover_oneway'].std():.2%}")
    print(f"  Min:     {t['turnover_oneway'].min():.2%}")
    print(f"  Max:     {t['turnover_oneway'].max():.2%}")
    
    print(f"\n--- Raw Turnover (before capping) ---")
    print(f"  Mean:    {t['turnover_raw'].mean():.2%}")
    print(f"  Max:     {t['turnover_raw'].max():.2%}")
    
    pct_capped = t['turnover_capped'].sum() / len(t) * 100
    print(f"  Periods where turnover was capped: {pct_capped:.1f}%")
    
    print(f"\n--- Annualized Turnover ---")
    # With 5-day rebalance, ~50 rebalances per year
    periods_per_year = 252 / 5
    ann_turnover = t['turnover_oneway'].mean() * periods_per_year
    print(f"  Estimated Annual Turnover: {ann_turnover:.0%}")
    print(f"  (This means ~{ann_turnover:.1f}x portfolio value traded per year)")
    
    print(f"\n--- Position Counts ---")
    print(f"  Avg Positions Held: {t['n_positions'].mean():.1f}")
    print(f"  Avg Position Size:  {t['avg_position_size'].mean():.2%}")
    print(f"  Max Position Size:  {t['max_position_size'].max():.2%}")


def print_cost_analysis(perf_df: pd.DataFrame, breakeven: float) -> None:
    """Print cost analysis results."""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE: GROSS vs NET (after transaction costs)")
    print("=" * 70)
    
    print("\n{:<25} {:>10} {:>12} {:>10} {:>10} {:>12}".format(
        "Model", "Cost(bps)", "Ann Return", "Ann Vol", "Sharpe", "Total Ret"))
    print("-" * 70)
    
    for _, row in perf_df.iterrows():
        print("{:<25} {:>10.0f} {:>11.1%} {:>10.1%} {:>10.2f} {:>11.1%}".format(
            row["Model"],
            row["Cost_bps"],
            row["Ann_Return"],
            row["Ann_Vol"],
            row["Sharpe"],
            row["Total_Return"],
        ))
    
    print("\n" + "-" * 70)
    print(f"BREAKEVEN COST: {breakeven * 10000:.1f} bps roundtrip")
    print(f"  (Strategy becomes unprofitable above this cost level)")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    gross_sharpe = perf_df[perf_df["Model"] == "GROSS (no costs)"]["Sharpe"].values[0]
    inst_sharpe = perf_df[perf_df["Model"] == "B_Institutional"]["Sharpe"].values[0]
    
    print(f"""
Cost Model Assumptions:
  A_Retail (10bps):       Low-cost broker, very liquid stocks only
  B_Institutional (20bps): Typical institutional execution with some impact
  C_Conservative (40bps):  Higher impact, less liquid names in portfolio
  D_HighImpact (60bps):    Significant capacity constraints

Your Strategy:
  Gross Sharpe: {gross_sharpe:.2f}
  Net Sharpe (Institutional): {inst_sharpe:.2f}
  Sharpe Degradation: {(gross_sharpe - inst_sharpe):.2f} ({(gross_sharpe - inst_sharpe)/gross_sharpe*100:.0f}% of gross)
  
Breakeven at {breakeven*10000:.0f}bps suggests the strategy can tolerate:
  - Up to {breakeven*10000/2:.0f}bps one-way costs (commission + half-spread + impact)
  - This is {"GOOD" if breakeven > 0.003 else "CONCERNING"} for institutional execution
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze turnover and transaction costs")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--alpha-column", default=DEFAULT_CONFIG["alpha_column"],
                        help=f"Alpha column (default: {DEFAULT_CONFIG['alpha_column']})")
    parser.add_argument("--start-date", default=DEFAULT_CONFIG["start_date"])
    parser.add_argument("--end-date", default=DEFAULT_CONFIG["end_date"])
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config["alpha_column"] = args.alpha_column
    config["start_date"] = args.start_date
    config["end_date"] = args.end_date
    
    print("\n" + "=" * 70)
    print("KAIROS TURNOVER & TRANSACTION COST ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Alpha Column:    {config['alpha_column']}")
    print(f"  Date Range:      {config['start_date']} to {config['end_date']}")
    print(f"  Top N:           {config['top_n']}")
    print(f"  Rebalance Every: {config['rebalance_every']} days")
    print(f"  Target Vol:      {config['target_vol']:.0%}")
    print(f"  Turnover Cap:    {config['turnover_cap']:.0%} per rebalance")
    
    # Connect and load data
    con = duckdb.connect(args.db, read_only=True)
    
    try:
        df = load_data(con, config)
    except ValueError as e:
        logger.error(str(e))
        con.close()
        return
    
    con.close()
    
    # Run backtest with turnover tracking
    bt_df, turnover_df = run_backtest_with_turnover(df, config)
    
    if bt_df.empty:
        logger.error("Backtest produced no results")
        return
    
    # Print turnover summary
    print_turnover_summary(turnover_df)
    
    # Calculate performance under different cost models
    perf_df = calculate_net_performance(bt_df, COST_MODELS, config)
    
    # Calculate breakeven cost
    breakeven = calculate_breakeven_cost(bt_df, config)
    
    # Print cost analysis
    print_cost_analysis(perf_df, breakeven)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()