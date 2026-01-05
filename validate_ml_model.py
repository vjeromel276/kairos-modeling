#!/usr/bin/env python3
"""
validate_ml_model.py
====================
Extensible validation framework for ML alpha signals.

Validates that the ML model is not overfit by checking:
1. Monthly rolling IC - signal stability over time
2. Quintile spread analysis - monotonic return relationship

Usage:
    python validate_ml_model.py --db data/kairos.duckdb
    python validate_ml_model.py --db data/kairos.duckdb --alpha-column alpha_ml_v2_tuned_clf
    python validate_ml_model.py --db data/kairos.duckdb --start-date 2015-01-01 --end-date 2024-12-31
"""

import argparse
import logging
import warnings
from datetime import datetime

import duckdb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Suppress pandas FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "alpha_column": "alpha_ml_v2_tuned_clf",
    "return_column": "ret_5d_f",
    "min_adv": 50_000_000,      # $50M ADV filter
    "min_price": 10.0,          # $10 minimum price
    "start_date": "2015-01-01",
    "end_date": None,           # None = latest available
    "n_quintiles": 5,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_data(con: duckdb.DuckDBPyConnection, config: dict) -> pd.DataFrame:
    """
    Load data for validation with production universe filters.
    """
    logger.info("Loading validation data...")
    
    alpha_col = config["alpha_column"]
    return_col = config["return_column"]
    min_adv = config["min_adv"]
    min_price = config["min_price"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    
    where_clauses = [
        f"m.{alpha_col} IS NOT NULL",
        f"m.{return_col} IS NOT NULL",
        f"m.adv_20 > {min_adv}",
        f"s.close > {min_price}",
        "t.\"table\" = 'SEP'",
    ]
    
    if start_date:
        where_clauses.append(f"m.date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"m.date <= '{end_date}'")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
        SELECT 
            m.ticker,
            m.date,
            m.{alpha_col} as alpha,
            m.{return_col} as actual_return,
            t.sector
        FROM feat_matrix_v2 m
        JOIN tickers t ON m.ticker = t.ticker
        JOIN sep_base_academic s ON m.ticker = s.ticker AND m.date = s.date
        WHERE {where_sql}
        ORDER BY m.date, m.ticker
    """
    
    df = con.execute(query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Unique dates: {df['date'].nunique():,}")
    logger.info(f"  Unique tickers: {df['ticker'].nunique():,}")
    
    return df


# =============================================================================
# VALIDATION 1: MONTHLY ROLLING IC
# =============================================================================

def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Information Coefficient (Spearman correlation)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 30:
        return np.nan
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def validate_monthly_rolling_ic(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate IC for each month."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION 1: MONTHLY ROLLING IC")
    logger.info("=" * 60)
    
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    
    results = []
    for ym, group in df.groupby('year_month'):
        ic = calculate_ic(group['actual_return'].values, group['alpha'].values)
        results.append({
            'year_month': str(ym),
            'ic': ic,
            'n_obs': len(group),
        })
    
    results_df = pd.DataFrame(results)
    valid_ics = results_df['ic'].dropna()
    
    logger.info(f"\nMonthly IC Summary:")
    logger.info(f"  Months analyzed: {len(results_df)}")
    logger.info(f"  Months with valid IC: {len(valid_ics)}")
    logger.info(f"  Mean IC: {valid_ics.mean():.4f}")
    logger.info(f"  Std IC: {valid_ics.std():.4f}")
    logger.info(f"  Min IC: {valid_ics.min():.4f}")
    logger.info(f"  Max IC: {valid_ics.max():.4f}")
    logger.info(f"  % Positive: {(valid_ics > 0).mean() * 100:.1f}%")
    
    # Flag concerning patterns
    negative_streak = 0
    max_negative_streak = 0
    for ic in valid_ics:
        if ic < 0:
            negative_streak += 1
            max_negative_streak = max(max_negative_streak, negative_streak)
        else:
            negative_streak = 0
    
    logger.info(f"  Max consecutive negative months: {max_negative_streak}")
    
    # Yearly breakdown
    results_df['year'] = results_df['year_month'].str[:4]
    yearly = results_df.groupby('year').agg({
        'ic': ['mean', 'std', 'count'],
        'n_obs': 'sum'
    }).round(4)
    yearly.columns = ['mean_ic', 'std_ic', 'n_months', 'n_obs']
    
    logger.info(f"\nIC by Year:")
    logger.info(f"{'Year':<6} {'Mean IC':>10} {'Std IC':>10} {'Months':>8} {'N Obs':>12}")
    logger.info("-" * 50)
    for year, row in yearly.iterrows():
        logger.info(f"{year:<6} {row['mean_ic']:>+10.4f} {row['std_ic']:>10.4f} {int(row['n_months']):>8} {int(row['n_obs']):>12,}")
    
    return results_df


# =============================================================================
# VALIDATION 2: QUINTILE SPREAD ANALYSIS
# =============================================================================

def validate_quintile_spread(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Analyze returns by alpha quintile.
    
    Q1 = TOP 20% (highest alpha) - these are your picks
    Q5 = BOTTOM 20% (lowest alpha) - stocks you avoid
    
    Good model: Q1 returns > Q2 > Q3 > Q4 > Q5 (monotonic)
    """
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION 2: QUINTILE SPREAD ANALYSIS")
    logger.info("=" * 60)
    
    n_quintiles = config.get("n_quintiles", 5)
    df = df.copy()
    
    # Assign quintiles: Q1 = highest alpha, Q5 = lowest alpha
    # Using negative alpha so qcut puts highest values in Q1
    df['quintile'] = df.groupby('date')['alpha'].transform(
        lambda x: pd.qcut(-x.rank(method='first'), q=n_quintiles, 
                          labels=[f'Q{i}' for i in range(1, n_quintiles + 1)])
    )
    
    # Calculate mean return by quintile and date
    quintile_returns = df.groupby(['date', 'quintile'], observed=True)['actual_return'].mean().unstack()
    
    # Overall quintile summary
    overall = quintile_returns.mean()
    
    logger.info(f"\nOverall Quintile Returns (annualized ~52x for weekly):")
    logger.info(f"{'Quintile':<12} {'Description':<20} {'Mean Return':>12} {'Annualized':>12}")
    logger.info("-" * 60)
    
    descriptions = {
        'Q1': 'TOP 20% (picks)',
        'Q2': 'Next 20%',
        'Q3': 'Middle 20%',
        'Q4': 'Low 20%',
        'Q5': 'BOTTOM 20% (avoid)',
    }
    
    for q in overall.index:
        mean_ret = overall[q]
        ann_ret = mean_ret * 52
        desc = descriptions.get(q, '')
        logger.info(f"{q:<12} {desc:<20} {mean_ret:>+12.4%} {ann_ret:>+12.1%}")
    
    # Spread analysis
    q1_return = overall['Q1']
    q5_return = overall[f'Q{n_quintiles}']
    spread = q1_return - q5_return
    
    logger.info(f"\nQ1 - Q{n_quintiles} Spread (Top minus Bottom):")
    logger.info(f"  Weekly:     {spread:+.4%}")
    logger.info(f"  Annualized: {spread * 52:+.1%}")
    
    # Check monotonicity
    is_monotonic = all(overall.iloc[i] >= overall.iloc[i+1] for i in range(len(overall)-1))
    logger.info(f"\nMonotonic (Q1 > Q2 > Q3 > Q4 > Q5): {'YES ✓' if is_monotonic else 'NO ✗'}")
    
    # Yearly quintile analysis
    df['year'] = df['date'].dt.year
    yearly_quintile = df.groupby(['year', 'quintile'], observed=True)['actual_return'].mean().unstack()
    
    logger.info(f"\nQuintile Returns by Year:")
    header = f"{'Year':<6}" + "".join([f"{'Q'+str(i):>10}" for i in range(1, n_quintiles + 1)]) + f"{'Spread':>12}"
    logger.info(header)
    logger.info("-" * (6 + 10 * n_quintiles + 12))
    
    for year in sorted(yearly_quintile.index):
        row = yearly_quintile.loc[year]
        spread_yr = row['Q1'] - row[f'Q{n_quintiles}']
        values = "".join([f"{row[f'Q{i}']:>+10.3%}" for i in range(1, n_quintiles + 1)])
        logger.info(f"{year:<6}{values}{spread_yr:>+12.3%}")
    
    # Check for problem years (negative spread)
    yearly_spreads = yearly_quintile['Q1'] - yearly_quintile[f'Q{n_quintiles}']
    negative_years = yearly_spreads[yearly_spreads < 0].index.tolist()
    positive_years = yearly_spreads[yearly_spreads > 0].index.tolist()
    
    if negative_years:
        logger.warning(f"\n⚠ Years with NEGATIVE spread (bottom beat top): {negative_years}")
    
    logger.info(f"\n✓ Years with POSITIVE spread: {len(positive_years)}/{len(yearly_spreads)}")
    
    return quintile_returns


# =============================================================================
# VALIDATION 3: IC BY MARKET REGIME (PLACEHOLDER)
# =============================================================================

def validate_by_regime(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Analyze IC performance across different market regimes.
    
    Regimes analyzed:
    1. Up months vs Down months (based on median stock return)
    2. High volatility vs Low volatility months
    """
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION 3: IC BY MARKET REGIME")
    logger.info("=" * 60)
    
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate monthly market stats
    monthly_stats = df.groupby('year_month').agg({
        'actual_return': ['mean', 'std'],
        'alpha': 'count'
    }).reset_index()
    monthly_stats.columns = ['year_month', 'mkt_return', 'mkt_vol', 'n_obs']
    
    # Classify months
    median_return = monthly_stats['mkt_return'].median()
    median_vol = monthly_stats['mkt_vol'].median()
    
    monthly_stats['regime_direction'] = np.where(
        monthly_stats['mkt_return'] >= median_return, 'UP', 'DOWN'
    )
    monthly_stats['regime_vol'] = np.where(
        monthly_stats['mkt_vol'] >= median_vol, 'HIGH_VOL', 'LOW_VOL'
    )
    
    # Calculate IC for each month
    monthly_ic = []
    for ym, group in df.groupby('year_month'):
        ic = calculate_ic(group['actual_return'].values, group['alpha'].values)
        monthly_ic.append({'year_month': ym, 'ic': ic})
    
    ic_df = pd.DataFrame(monthly_ic)
    ic_df = ic_df.merge(monthly_stats, on='year_month')
    
    # === Analysis by Market Direction ===
    logger.info("\n3a. IC BY MARKET DIRECTION")
    logger.info("-" * 40)
    
    direction_analysis = ic_df.groupby('regime_direction').agg({
        'ic': ['mean', 'std', 'count'],
        'mkt_return': 'mean'
    }).round(4)
    direction_analysis.columns = ['mean_ic', 'std_ic', 'n_months', 'avg_mkt_ret']
    
    for regime, row in direction_analysis.iterrows():
        pct_positive = (ic_df[ic_df['regime_direction'] == regime]['ic'] > 0).mean() * 100
        logger.info(f"  {regime} months ({int(row['n_months'])} months):")
        logger.info(f"    Mean IC: {row['mean_ic']:+.4f}")
        logger.info(f"    IC Std:  {row['std_ic']:.4f}")
        logger.info(f"    % Positive IC: {pct_positive:.0f}%")
        logger.info(f"    Avg Market Return: {row['avg_mkt_ret']*100:+.2f}%")
    
    # === Analysis by Volatility Regime ===
    logger.info("\n3b. IC BY VOLATILITY REGIME")
    logger.info("-" * 40)
    
    vol_analysis = ic_df.groupby('regime_vol').agg({
        'ic': ['mean', 'std', 'count'],
        'mkt_vol': 'mean'
    }).round(4)
    vol_analysis.columns = ['mean_ic', 'std_ic', 'n_months', 'avg_vol']
    
    for regime, row in vol_analysis.iterrows():
        pct_positive = (ic_df[ic_df['regime_vol'] == regime]['ic'] > 0).mean() * 100
        logger.info(f"  {regime} months ({int(row['n_months'])} months):")
        logger.info(f"    Mean IC: {row['mean_ic']:+.4f}")
        logger.info(f"    IC Std:  {row['std_ic']:.4f}")
        logger.info(f"    % Positive IC: {pct_positive:.0f}%")
        logger.info(f"    Avg Cross-Sectional Vol: {row['avg_vol']*100:.2f}%")
    
    # === Combined Regime Analysis ===
    logger.info("\n3c. IC BY COMBINED REGIME")
    logger.info("-" * 40)
    
    ic_df['combined_regime'] = ic_df['regime_direction'] + '_' + ic_df['regime_vol']
    
    combined = ic_df.groupby('combined_regime').agg({
        'ic': ['mean', 'count']
    }).round(4)
    combined.columns = ['mean_ic', 'n_months']
    combined = combined.sort_values('mean_ic', ascending=False)
    
    logger.info(f"  {'Regime':<20} {'Mean IC':>10} {'Months':>8}")
    logger.info("  " + "-" * 40)
    for regime, row in combined.iterrows():
        logger.info(f"  {regime:<20} {row['mean_ic']:>+10.4f} {int(row['n_months']):>8}")
    
    # === Key Insight ===
    logger.info("\n3d. KEY INSIGHT")
    logger.info("-" * 40)
    
    up_ic = direction_analysis.loc['UP', 'mean_ic']
    down_ic = direction_analysis.loc['DOWN', 'mean_ic']
    high_vol_ic = vol_analysis.loc['HIGH_VOL', 'mean_ic']
    low_vol_ic = vol_analysis.loc['LOW_VOL', 'mean_ic']
    
    if down_ic > 0:
        logger.info(f"  ✓ Signal works in DOWN markets (IC={down_ic:+.4f})")
    else:
        logger.warning(f"  ⚠ Signal struggles in DOWN markets (IC={down_ic:+.4f})")
    
    if high_vol_ic > 0:
        logger.info(f"  ✓ Signal works in HIGH volatility (IC={high_vol_ic:+.4f})")
    else:
        logger.warning(f"  ⚠ Signal struggles in HIGH volatility (IC={high_vol_ic:+.4f})")
    
    # Check for regime dependency
    ic_gap_direction = abs(up_ic - down_ic)
    ic_gap_vol = abs(high_vol_ic - low_vol_ic)
    
    if ic_gap_direction > 0.03:
        logger.warning(f"  ⚠ Large IC gap between UP/DOWN markets ({ic_gap_direction:.4f})")
    else:
        logger.info(f"  ✓ IC relatively stable across UP/DOWN markets (gap={ic_gap_direction:.4f})")
    
    if ic_gap_vol > 0.03:
        logger.warning(f"  ⚠ Large IC gap between HIGH/LOW vol ({ic_gap_vol:.4f})")
    else:
        logger.info(f"  ✓ IC relatively stable across vol regimes (gap={ic_gap_vol:.4f})")
    
    return ic_df


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(
    monthly_ic: pd.DataFrame,
    quintile_returns: pd.DataFrame,
    config: dict
):
    """Generate overall validation summary."""
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY REPORT")
    logger.info("=" * 60)
    
    valid_ics = monthly_ic['ic'].dropna()
    mean_ic = valid_ics.mean()
    pct_positive = (valid_ics > 0).mean() * 100
    
    overall_quintile = quintile_returns.mean()
    spread = overall_quintile['Q1'] - overall_quintile['Q5']
    is_monotonic = all(overall_quintile.iloc[i] >= overall_quintile.iloc[i+1] 
                       for i in range(len(overall_quintile)-1))
    
    logger.info("\nVERDICT:")
    logger.info("-" * 40)
    
    issues = []
    positives = []
    
    # IC checks
    if mean_ic >= 0.05:
        positives.append(f"Strong mean IC ({mean_ic:.4f})")
    elif mean_ic >= 0.02:
        positives.append(f"Acceptable mean IC ({mean_ic:.4f})")
    else:
        issues.append(f"Low mean IC ({mean_ic:.4f} < 0.02)")
    
    if pct_positive >= 70:
        positives.append(f"High % positive months ({pct_positive:.0f}%)")
    elif pct_positive < 60:
        issues.append(f"Too many negative months ({100-pct_positive:.0f}%)")
    
    if valid_ics.std() > 0.05:
        issues.append(f"High IC volatility (std={valid_ics.std():.4f})")
    
    # Quintile checks
    if spread > 0:
        positives.append(f"Positive Q1-Q5 spread ({spread:.4%} weekly)")
    else:
        issues.append(f"Negative Q1-Q5 spread ({spread:.4%})")
    
    if is_monotonic:
        positives.append("Monotonic quintile returns")
    else:
        issues.append("Non-monotonic quintile returns")
    
    if positives:
        logger.info("✓ STRENGTHS:")
        for p in positives:
            logger.info(f"  + {p}")
    
    if issues:
        logger.warning("\n⚠ CONCERNS:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ All validation checks PASSED")
    
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Mean IC:           {mean_ic:+.4f}")
    logger.info(f"  IC Std:            {valid_ics.std():.4f}")
    logger.info(f"  % Months Positive: {pct_positive:.1f}%")
    logger.info(f"  Q1-Q5 Spread:      {spread:+.4%} weekly / {spread*52:+.1%} annual")
    logger.info(f"  Monotonic:         {'Yes' if is_monotonic else 'No'}")
    
    # Overall assessment
    logger.info("\n" + "-" * 40)
    if len(issues) == 0:
        logger.info("OVERALL: Model validation PASSED ✓")
    elif len(issues) <= 1 and spread > 0:
        logger.info("OVERALL: Model validation ACCEPTABLE with minor concerns")
    else:
        logger.warning("OVERALL: Model validation has SIGNIFICANT CONCERNS")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate ML alpha model")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--alpha-column", default=DEFAULT_CONFIG["alpha_column"])
    parser.add_argument("--return-column", default=DEFAULT_CONFIG["return_column"])
    parser.add_argument("--min-adv", type=float, default=DEFAULT_CONFIG["min_adv"])
    parser.add_argument("--min-price", type=float, default=DEFAULT_CONFIG["min_price"])
    parser.add_argument("--start-date", default=DEFAULT_CONFIG["start_date"])
    parser.add_argument("--end-date", default=DEFAULT_CONFIG["end_date"])
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    
    config = {
        "alpha_column": args.alpha_column,
        "return_column": args.return_column,
        "min_adv": args.min_adv,
        "min_price": args.min_price,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_quintiles": 5,
    }
    
    logger.info("=" * 60)
    logger.info("ML MODEL VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db}")
    logger.info(f"Alpha column: {config['alpha_column']}")
    logger.info(f"Return column: {config['return_column']}")
    logger.info(f"Filters: ADV > ${config['min_adv']/1e6:.0f}M, Price > ${config['min_price']}")
    logger.info(f"Date range: {config['start_date']} to {config['end_date'] or 'latest'}")
    
    con = duckdb.connect(args.db, read_only=True)
    
    try:
        df = load_validation_data(con, config)
        
        if len(df) == 0:
            logger.error("No data loaded - check filters and column names")
            return
        
        monthly_ic = validate_monthly_rolling_ic(df, config)
        quintile_returns = validate_quintile_spread(df, config)
        validate_by_regime(df, config)
        generate_summary_report(monthly_ic, quintile_returns, config)
        
        if args.output_csv:
            monthly_ic.to_csv(args.output_csv, index=False)
            logger.info(f"\nSaved monthly IC to: {args.output_csv}")
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 60)
        
    finally:
        con.close()


if __name__ == "__main__":
    main()