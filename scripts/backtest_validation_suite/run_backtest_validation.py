#!/usr/bin/env python3
"""
run_backtest_validation.py

Unified backtest and validation framework with YAML configuration.
Combines Risk4 portfolio construction with comprehensive signal validation.

Usage:
    python run_backtest_validation.py --config configs/tight_50m.yaml --db data/kairos.duckdb

Outputs:
    results/{timestamp}_{config_name}/
        ├── config_used.yaml
        ├── universe_stats.csv
        ├── decile_analysis.csv
        ├── quintile_analysis.csv
        ├── ic_monthly.csv
        ├── ic_regime.csv
        ├── backtest_returns.csv
        └── summary.json

Author: Kairos Quant Engineering
Version: 1.0
"""

import argparse
import json
import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults for optional fields
    defaults = {
        'filters': {
            'adv_min': 2_000_000,
            'price_min': 0.0,
        },
        'alpha': {
            'column': 'alpha_ml_v2_tuned_clf',
            'target': 'ret_5d_f',
        },
        'portfolio': {
            'top_n': 75,
            'target_vol': 0.25,
            'max_position_pct': 0.03,
            'max_sector_mult': 2.0,
        },
        'turnover': {
            'lambda_tc': 0.20,
            'turnover_cap': 0.20,
        },
        'backtest': {
            'start_date': '2015-01-01',
            'end_date': '2024-12-31',
            'rebalance_every': 5,
        },
    }
    
    # Merge defaults with provided config
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = section_defaults
        else:
            for key, value in section_defaults.items():
                if key not in config[section]:
                    config[section][key] = value
    
    return config


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(con, config: dict) -> pd.DataFrame:
    """Load data with configurable filters."""
    alpha_col = config['alpha']['column']
    target_col = config['alpha']['target']
    adv_min = config['filters']['adv_min']
    price_min = config['filters']['price_min']
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    
    logger.info(f"Loading data with filters: ADV >= ${adv_min:,.0f}, Price >= ${price_min:.2f}")
    
    where_clauses = [
        f"fm.{alpha_col} IS NOT NULL",
        f"fm.{target_col} IS NOT NULL",
        "fm.vol_blend IS NOT NULL",
        "fm.vol_blend > 0",
        "fm.adv_20 IS NOT NULL",
        f"fm.adv_20 >= {adv_min}",
        "t.sector IS NOT NULL",
        "t.\"table\" = 'SEP'",
    ]
    
    if price_min > 0:
        where_clauses.append(f"s.close >= {price_min}")
    
    if start_date:
        where_clauses.append(f"fm.date >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"fm.date <= DATE '{end_date}'")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
        SELECT
            fm.ticker,
            fm.date,
            fm.{alpha_col} AS alpha,
            fm.{target_col} AS target,
            fm.vol_blend,
            fm.adv_20,
            t.sector,
            s.close AS price
        FROM feat_matrix_v2 fm
        JOIN tickers t ON fm.ticker = t.ticker
        JOIN sep_base_academic s ON fm.ticker = s.ticker AND fm.date = s.date
        WHERE {where_sql}
        ORDER BY fm.date, fm.ticker
    """
    
    df = con.execute(query).fetchdf()
    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates, {df['ticker'].nunique()} unique tickers")
    
    return df


# ============================================================================
# UNIVERSE STATISTICS
# ============================================================================

def compute_universe_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute universe size statistics by date."""
    stats = df.groupby('date').agg(
        n_stocks=('ticker', 'nunique'),
        n_sectors=('sector', 'nunique'),
        avg_adv=('adv_20', 'mean'),
        avg_price=('price', 'mean'),
        alpha_std=('alpha', 'std'),
    ).reset_index()
    
    return stats


# ============================================================================
# VALIDATION: DECILE ANALYSIS
# ============================================================================

def compute_decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute returns by decile."""
    logger.info("Computing decile analysis...")
    
    def assign_decile(x):
        """Assign deciles: 10 = highest alpha, 1 = lowest alpha."""
        return pd.qcut(-x.rank(method='first'), q=10, labels=range(10, 0, -1))
    
    df = df.copy()
    df['decile'] = df.groupby('date')['alpha'].transform(assign_decile)
    df['decile'] = df['decile'].astype(int)
    
    # Aggregate by decile
    decile_stats = df.groupby('decile').agg(
        n_obs=('target', 'count'),
        mean_ret=('target', 'mean'),
        std_ret=('target', 'std'),
    ).reset_index()
    
    # Annualize (assuming 5-day returns, ~52 periods/year)
    decile_stats['ann_ret'] = decile_stats['mean_ret'] * 52
    decile_stats['ann_vol'] = decile_stats['std_ret'] * np.sqrt(52)
    decile_stats['sharpe'] = decile_stats['ann_ret'] / decile_stats['ann_vol']
    
    # Check monotonicity
    returns_ordered = decile_stats.sort_values('decile', ascending=False)['ann_ret'].tolist()
    is_monotonic = all(returns_ordered[i] >= returns_ordered[i+1] for i in range(len(returns_ordered)-1))
    
    logger.info(f"Decile monotonicity: {'YES' if is_monotonic else 'NO'}")
    logger.info(f"D10 (top) return: {decile_stats[decile_stats['decile']==10]['ann_ret'].values[0]:.1%}")
    logger.info(f"D1 (bottom) return: {decile_stats[decile_stats['decile']==1]['ann_ret'].values[0]:.1%}")
    
    return decile_stats


def compute_decile_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute decile spreads by year."""
    def assign_decile(x):
        return pd.qcut(-x.rank(method='first'), q=10, labels=range(10, 0, -1))
    
    df = df.copy()
    df['decile'] = df.groupby('date')['alpha'].transform(assign_decile)
    df['decile'] = df['decile'].astype(int)
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    yearly = df.groupby(['year', 'decile'])['target'].mean().unstack()
    yearly['d10_d1_spread'] = yearly[10] - yearly[1]
    
    return yearly.reset_index()


# ============================================================================
# VALIDATION: QUINTILE ANALYSIS
# ============================================================================

def compute_quintile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute returns by quintile."""
    logger.info("Computing quintile analysis...")
    
    def assign_quintile(x):
        """Assign quintiles: Q1 = highest alpha, Q5 = lowest alpha."""
        return pd.qcut(-x.rank(method='first'), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    df = df.copy()
    df['quintile'] = df.groupby('date')['alpha'].transform(assign_quintile)
    
    # Aggregate by quintile
    quintile_stats = df.groupby('quintile').agg(
        n_obs=('target', 'count'),
        mean_ret=('target', 'mean'),
        std_ret=('target', 'std'),
    ).reset_index()
    
    # Annualize
    quintile_stats['ann_ret'] = quintile_stats['mean_ret'] * 52
    quintile_stats['ann_vol'] = quintile_stats['std_ret'] * np.sqrt(52)
    quintile_stats['sharpe'] = quintile_stats['ann_ret'] / quintile_stats['ann_vol']
    
    # Q1-Q5 spread
    q1_ret = quintile_stats[quintile_stats['quintile'] == 'Q1']['ann_ret'].values[0]
    q5_ret = quintile_stats[quintile_stats['quintile'] == 'Q5']['ann_ret'].values[0]
    spread = q1_ret - q5_ret
    
    logger.info(f"Q1 (top) return: {q1_ret:.1%}")
    logger.info(f"Q5 (bottom) return: {q5_ret:.1%}")
    logger.info(f"Q1-Q5 spread: {spread:.1%}")
    
    return quintile_stats


def compute_quintile_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quintile returns by year."""
    def assign_quintile(x):
        return pd.qcut(-x.rank(method='first'), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    df = df.copy()
    df['quintile'] = df.groupby('date')['alpha'].transform(assign_quintile)
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    yearly = df.groupby(['year', 'quintile'])['target'].mean().unstack()
    yearly['q1_q5_spread'] = yearly['Q1'] - yearly['Q5']
    
    return yearly.reset_index()


# ============================================================================
# VALIDATION: IC ANALYSIS
# ============================================================================

def compute_ic_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly rolling Information Coefficient."""
    logger.info("Computing monthly IC...")
    
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    def calc_ic(group):
        if len(group) < 30:
            return np.nan
        corr, _ = stats.spearmanr(group['alpha'], group['target'])
        return corr
    
    monthly_ic = df.groupby('year_month').apply(calc_ic).reset_index()
    monthly_ic.columns = ['year_month', 'ic']
    monthly_ic['year_month'] = monthly_ic['year_month'].astype(str)
    
    # Add year for aggregation
    monthly_ic['year'] = monthly_ic['year_month'].str[:4].astype(int)
    
    valid_ic = monthly_ic['ic'].dropna()
    logger.info(f"Mean IC: {valid_ic.mean():.4f}")
    logger.info(f"IC Std: {valid_ic.std():.4f}")
    logger.info(f"% Positive: {(valid_ic > 0).mean():.1%}")
    
    return monthly_ic


def compute_ic_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IC by market regime."""
    logger.info("Computing IC by regime...")
    
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Compute monthly market return and volatility
    monthly_stats = df.groupby('year_month').agg(
        market_ret=('target', 'mean'),
        cross_vol=('target', 'std'),
    ).reset_index()
    
    # Classify regimes
    monthly_stats['direction'] = np.where(monthly_stats['market_ret'] > 0, 'UP', 'DOWN')
    vol_median = monthly_stats['cross_vol'].median()
    monthly_stats['volatility'] = np.where(monthly_stats['cross_vol'] > vol_median, 'HIGH_VOL', 'LOW_VOL')
    monthly_stats['regime'] = monthly_stats['direction'] + '_' + monthly_stats['volatility']
    
    # Compute IC per month
    def calc_ic(group):
        if len(group) < 30:
            return np.nan
        corr, _ = stats.spearmanr(group['alpha'], group['target'])
        return corr
    
    monthly_ic = df.groupby('year_month').apply(calc_ic).reset_index()
    monthly_ic.columns = ['year_month', 'ic']
    
    # Merge with regime
    merged = monthly_ic.merge(monthly_stats[['year_month', 'direction', 'volatility', 'regime']], on='year_month')
    
    # Aggregate by regime
    regime_stats = merged.groupby('regime').agg(
        mean_ic=('ic', 'mean'),
        std_ic=('ic', 'std'),
        pct_positive=('ic', lambda x: (x > 0).mean()),
        n_months=('ic', 'count'),
    ).reset_index()
    
    # Also by direction and volatility separately
    direction_stats = merged.groupby('direction').agg(
        mean_ic=('ic', 'mean'),
        pct_positive=('ic', lambda x: (x > 0).mean()),
        n_months=('ic', 'count'),
    ).reset_index()
    
    vol_stats = merged.groupby('volatility').agg(
        mean_ic=('ic', 'mean'),
        pct_positive=('ic', lambda x: (x > 0).mean()),
        n_months=('ic', 'count'),
    ).reset_index()
    
    logger.info("IC by direction:")
    for _, row in direction_stats.iterrows():
        logger.info(f"  {row['direction']}: IC={row['mean_ic']:.4f}, {row['pct_positive']:.0%} positive")
    
    return regime_stats


# ============================================================================
# BACKTEST: RISK4 PORTFOLIO CONSTRUCTION
# ============================================================================

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


def annualize(returns: pd.Series, period_days: int = 5) -> tuple:
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


def run_backtest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run Risk4 backtest with configurable parameters.
    """
    top_n = config['portfolio']['top_n']
    target_vol = config['portfolio']['target_vol']
    max_stock_w = config['portfolio']['max_position_pct']
    sector_cap_mult = config['portfolio']['max_sector_mult']
    lambda_tc = config['turnover']['lambda_tc']
    turnover_cap = config['turnover']['turnover_cap']
    rebalance_every = config['backtest']['rebalance_every']
    
    logger.info(f"Running backtest: top_n={top_n}, target_vol={target_vol:.0%}, rebal_every={rebalance_every}")
    
    # Compute alpha z-scores
    df = df.copy()
    df['alpha_z'] = df.groupby('date')['alpha'].transform(safe_z)
    df['alpha_z'] = df['alpha_z'].clip(-3.0, 3.0)
    
    dates = sorted(df['date'].unique())
    rebal_dates = dates[::rebalance_every]
    
    records = []
    w_old = pd.Series(dtype=float)
    
    for d in rebal_dates:
        day = df[df['date'] == d].copy()
        if day.empty:
            continue
        
        universe = day.copy()
        
        # Select top N by alpha
        picks = day.sort_values('alpha_z', ascending=False).head(top_n).copy()
        if picks.empty:
            continue
        
        picks = picks.drop_duplicates(subset='ticker', keep='first')
        picks = picks.set_index('ticker')
        
        # Initial weights: alpha_z / vol_blend
        w_prop = picks['alpha_z'].clip(lower=0.0)
        if w_prop.sum() == 0:
            continue
        
        w_prop = w_prop / picks['vol_blend'].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w_prop.sum() == 0:
            continue
        
        w_prop = w_prop / w_prop.sum()
        picks['weight_prop'] = w_prop
        
        # Cap per-stock weights
        picks['weight_prop'] = picks['weight_prop'].clip(upper=max_stock_w)
        if picks['weight_prop'].sum() == 0:
            continue
        picks['weight_prop'] /= picks['weight_prop'].sum()
        
        # Sector caps
        sector_counts = universe.groupby('sector')['ticker'].count()
        total_univ = float(len(universe))
        sector_univ_w = sector_counts / total_univ
        
        sector_port_w = picks.groupby('sector')['weight_prop'].sum()
        caps = sector_univ_w * sector_cap_mult
        caps = caps.reindex(sector_port_w.index).fillna(0.0)
        
        for sec, w_sec in sector_port_w.items():
            cap = caps.get(sec, 0.0)
            if w_sec > cap and cap > 0:
                scale = cap / w_sec
                picks.loc[picks['sector'] == sec, 'weight_prop'] *= scale
        
        if picks['weight_prop'].sum() <= 0:
            continue
        picks['weight_prop'] /= picks['weight_prop'].sum()
        
        # Turnover control
        w_prop = picks['weight_prop'].copy()
        w_prop = w_prop[~w_prop.index.duplicated(keep='first')]
        
        if w_old.empty:
            w_new = w_prop.copy()
        else:
            all_tickers = pd.Index(w_old.index).union(w_prop.index).unique()
            w_old_full = w_old.reindex(all_tickers).fillna(0.0)
            w_prop_full = w_prop.reindex(all_tickers).fillna(0.0)
            
            w_pre = (1 - lambda_tc) * w_old_full + lambda_tc * w_prop_full
            turnover = float((w_pre - w_old_full).abs().sum())
            
            if turnover > turnover_cap:
                scale = turnover_cap / turnover
                w_new_full = w_old_full + scale * (w_pre - w_old_full)
            else:
                w_new_full = w_pre
            
            if w_new_full.sum() <= 0:
                w_new_full = w_old_full
            w_new_full = w_new_full / w_new_full.sum()
            
            w_new = w_new_full.reindex(w_prop.index).fillna(0.0)
        
        w_new = w_new[~w_new.index.duplicated(keep='first')]
        w_old = w_new.copy()
        
        # Compute returns
        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue
        
        port_ret = float((picks.loc[common, 'target'] * w_new.loc[common]).sum())
        bench_ret = float(universe['target'].mean())
        
        records.append({
            'date': d,
            'port_ret_raw': port_ret,
            'bench_ret_raw': bench_ret,
            'n_positions': len(common),
        })
    
    bt = pd.DataFrame.from_records(records).sort_values('date').reset_index(drop=True)
    if bt.empty:
        logger.warning("Empty backtest")
        return bt
    
    # Vol targeting
    raw = bt['port_ret_raw']
    _, raw_vol, _ = annualize(raw, rebalance_every)
    logger.info(f"Raw annualized vol: {raw_vol:.2%}")
    
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    logger.info(f"Vol scale factor: {scale:.4f}")
    
    bt['port_ret'] = bt['port_ret_raw'] * scale
    bt['bench_ret'] = bt['bench_ret_raw']
    bt['active_ret'] = bt['port_ret'] - bt['bench_ret']
    
    return bt


def compute_backtest_summary(bt: pd.DataFrame, config: dict) -> dict:
    """Compute summary statistics from backtest."""
    if bt.empty:
        return {}
    
    rebalance_every = config['backtest']['rebalance_every']
    target_vol = config['portfolio']['target_vol']
    
    p = bt['port_ret']
    b = bt['bench_ret']
    a = bt['active_ret']
    
    port_eq = (1.0 + p).cumprod()
    bench_eq = (1.0 + b).cumprod()
    
    port_total = port_eq.iloc[-1] - 1.0
    bench_total = bench_eq.iloc[-1] - 1.0
    
    port_ann, port_vol, port_sharpe = annualize(p, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize(b, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize(a, rebalance_every)
    
    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)
    
    return {
        'portfolio': {
            'total_return': port_total,
            'annual_return': port_ann,
            'annual_vol': port_vol,
            'sharpe': port_sharpe,
            'max_drawdown': port_dd,
        },
        'benchmark': {
            'total_return': bench_total,
            'annual_return': bench_ann,
            'annual_vol': bench_vol,
            'sharpe': bench_sharpe,
            'max_drawdown': bench_dd,
        },
        'active': {
            'annual_return': active_ann,
            'annual_vol': active_vol,
            'sharpe': active_sharpe,
        },
        'target_vol': target_vol,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def print_summary(
    config: dict,
    universe_stats: pd.DataFrame,
    decile_stats: pd.DataFrame,
    quintile_stats: pd.DataFrame,
    ic_monthly: pd.DataFrame,
    backtest_summary: dict,
):
    """Print comprehensive summary to console."""
    
    print("\n" + "=" * 70)
    print(f"BACKTEST VALIDATION RESULTS: {config.get('name', 'unnamed')}")
    print("=" * 70)
    
    # Config summary
    print(f"\nCONFIGURATION:")
    print(f"  ADV Min:      ${config['filters']['adv_min']:,.0f}")
    print(f"  Price Min:    ${config['filters']['price_min']:.2f}")
    print(f"  Alpha:        {config['alpha']['column']}")
    print(f"  Top N:        {config['portfolio']['top_n']}")
    print(f"  Target Vol:   {config['portfolio']['target_vol']:.0%}")
    
    # Universe stats
    print(f"\nUNIVERSE STATISTICS:")
    print(f"  Avg stocks/date: {universe_stats['n_stocks'].mean():.0f}")
    print(f"  Min stocks/date: {universe_stats['n_stocks'].min()}")
    print(f"  Max stocks/date: {universe_stats['n_stocks'].max()}")
    
    # Decile summary
    print(f"\nDECILE ANALYSIS:")
    d10 = decile_stats[decile_stats['decile'] == 10]['ann_ret'].values[0]
    d1 = decile_stats[decile_stats['decile'] == 1]['ann_ret'].values[0]
    d10_sharpe = decile_stats[decile_stats['decile'] == 10]['sharpe'].values[0]
    d1_sharpe = decile_stats[decile_stats['decile'] == 1]['sharpe'].values[0]
    print(f"  D10 (top):    {d10:+.1%} (Sharpe: {d10_sharpe:.2f})")
    print(f"  D1 (bottom):  {d1:+.1%} (Sharpe: {d1_sharpe:.2f})")
    print(f"  D10-D1 spread: {d10-d1:+.1%}")
    
    # Check monotonicity
    returns_ordered = decile_stats.sort_values('decile', ascending=False)['ann_ret'].tolist()
    is_monotonic = all(returns_ordered[i] >= returns_ordered[i+1] for i in range(len(returns_ordered)-1))
    print(f"  Monotonic:    {'YES ✓' if is_monotonic else 'NO ✗'}")
    
    # Quintile summary
    print(f"\nQUINTILE ANALYSIS:")
    q1 = quintile_stats[quintile_stats['quintile'] == 'Q1']['ann_ret'].values[0]
    q5 = quintile_stats[quintile_stats['quintile'] == 'Q5']['ann_ret'].values[0]
    print(f"  Q1 (top 20%):    {q1:+.1%}")
    print(f"  Q5 (bottom 20%): {q5:+.1%}")
    print(f"  Q1-Q5 spread:    {q1-q5:+.1%}")
    
    # IC summary
    print(f"\nIC ANALYSIS:")
    valid_ic = ic_monthly['ic'].dropna()
    print(f"  Mean IC:      {valid_ic.mean():.4f}")
    print(f"  IC Std:       {valid_ic.std():.4f}")
    print(f"  % Positive:   {(valid_ic > 0).mean():.0%}")
    
    # Backtest summary
    if backtest_summary:
        print(f"\nBACKTEST RESULTS (Risk4):")
        print(f"  Portfolio Return:  {backtest_summary['portfolio']['annual_return']:+.1%}")
        print(f"  Portfolio Vol:     {backtest_summary['portfolio']['annual_vol']:.1%}")
        print(f"  Portfolio Sharpe:  {backtest_summary['portfolio']['sharpe']:.2f}")
        print(f"  Portfolio Max DD:  {backtest_summary['portfolio']['max_drawdown']:.1%}")
        print(f"  ---")
        print(f"  Benchmark Return:  {backtest_summary['benchmark']['annual_return']:+.1%}")
        print(f"  Benchmark Sharpe:  {backtest_summary['benchmark']['sharpe']:.2f}")
        print(f"  ---")
        print(f"  Active Return:     {backtest_summary['active']['annual_return']:+.1%}")
        print(f"  Active Sharpe:     {backtest_summary['active']['sharpe']:.2f}")
    
    print("\n" + "=" * 70)


def save_results(
    output_dir: Path,
    config: dict,
    config_path: str,
    universe_stats: pd.DataFrame,
    decile_stats: pd.DataFrame,
    decile_yearly: pd.DataFrame,
    quintile_stats: pd.DataFrame,
    quintile_yearly: pd.DataFrame,
    ic_monthly: pd.DataFrame,
    ic_regime: pd.DataFrame,
    backtest_returns: pd.DataFrame,
    backtest_summary: dict,
):
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config
    shutil.copy(config_path, output_dir / 'config_used.yaml')
    
    # Save CSVs
    universe_stats.to_csv(output_dir / 'universe_stats.csv', index=False)
    decile_stats.to_csv(output_dir / 'decile_analysis.csv', index=False)
    decile_yearly.to_csv(output_dir / 'decile_yearly.csv', index=False)
    quintile_stats.to_csv(output_dir / 'quintile_analysis.csv', index=False)
    quintile_yearly.to_csv(output_dir / 'quintile_yearly.csv', index=False)
    ic_monthly.to_csv(output_dir / 'ic_monthly.csv', index=False)
    ic_regime.to_csv(output_dir / 'ic_regime.csv', index=False)
    backtest_returns.to_csv(output_dir / 'backtest_returns.csv', index=False)
    
    # Compute validation metrics for summary
    valid_ic = ic_monthly['ic'].dropna()
    returns_ordered = decile_stats.sort_values('decile', ascending=False)['ann_ret'].tolist()
    is_monotonic = all(returns_ordered[i] >= returns_ordered[i+1] for i in range(len(returns_ordered)-1))
    
    # Build summary JSON
    summary = {
        'config_name': config.get('name', 'unnamed'),
        'generated_at': datetime.now().isoformat(),
        'filters': config['filters'],
        'universe': {
            'avg_stocks': float(universe_stats['n_stocks'].mean()),
            'min_stocks': int(universe_stats['n_stocks'].min()),
            'max_stocks': int(universe_stats['n_stocks'].max()),
        },
        'validation': {
            'decile_10_return': float(decile_stats[decile_stats['decile'] == 10]['ann_ret'].values[0]),
            'decile_1_return': float(decile_stats[decile_stats['decile'] == 1]['ann_ret'].values[0]),
            'decile_spread': float(decile_stats[decile_stats['decile'] == 10]['ann_ret'].values[0] - 
                                   decile_stats[decile_stats['decile'] == 1]['ann_ret'].values[0]),
            'decile_monotonic': is_monotonic,
            'quintile_q1_return': float(quintile_stats[quintile_stats['quintile'] == 'Q1']['ann_ret'].values[0]),
            'quintile_q5_return': float(quintile_stats[quintile_stats['quintile'] == 'Q5']['ann_ret'].values[0]),
            'quintile_spread': float(quintile_stats[quintile_stats['quintile'] == 'Q1']['ann_ret'].values[0] - 
                                     quintile_stats[quintile_stats['quintile'] == 'Q5']['ann_ret'].values[0]),
            'mean_ic': float(valid_ic.mean()),
            'ic_std': float(valid_ic.std()),
            'ic_pct_positive': float((valid_ic > 0).mean()),
        },
        'backtest': backtest_summary,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run backtest with validation')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--db', required=True, help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='results', help='Base output directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config_name = config.get('name', 'unnamed')
    
    logger.info(f"Running backtest validation: {config_name}")
    logger.info(f"Config: {args.config}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{timestamp}_{config_name}"
    
    # Connect to database
    con = duckdb.connect(args.db, read_only=True)
    
    # Load data
    df = load_data(con, config)
    
    if df.empty:
        logger.error("No data loaded. Check filters.")
        return 1
    
    # Run validation
    universe_stats = compute_universe_stats(df)
    decile_stats = compute_decile_analysis(df)
    decile_yearly = compute_decile_by_year(df)
    quintile_stats = compute_quintile_analysis(df)
    quintile_yearly = compute_quintile_by_year(df)
    ic_monthly = compute_ic_monthly(df)
    ic_regime = compute_ic_regime(df)
    
    # Run backtest
    backtest_returns = run_backtest(df, config)
    backtest_summary = compute_backtest_summary(backtest_returns, config)
    
    # Print summary
    print_summary(
        config, universe_stats, decile_stats, quintile_stats, ic_monthly, backtest_summary
    )
    
    # Save results
    save_results(
        output_dir, config, args.config,
        universe_stats, decile_stats, decile_yearly,
        quintile_stats, quintile_yearly,
        ic_monthly, ic_regime,
        backtest_returns, backtest_summary,
    )
    
    con.close()
    return 0


if __name__ == "__main__":
    exit(main())