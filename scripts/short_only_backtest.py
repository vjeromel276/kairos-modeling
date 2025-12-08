#!/usr/bin/env python3
"""
Short-Only Backtest for Kairos

This simulates ONLY the short book to isolate its performance.
We short the bottom N stocks by alpha and measure returns.

If this is negative (stocks go up), it explains why L/S underperforms.
"""

import argparse
import duckdb
import pandas as pd
import numpy as np

def compute_metrics(returns: pd.Series):
    """Compute standard performance metrics."""
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        'total_ret': total_ret * 100,
        'ann_ret': ann_ret * 100,
        'ann_vol': ann_vol * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    parser.add_argument("--alpha-column", type=str, default="alpha_composite_v33_regime")
    parser.add_argument("--target-column", type=str, default="ret_5d_f")
    parser.add_argument("--top-n-short", type=int, default=75, help="Number of stocks to short")
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="2025-11-28")
    parser.add_argument("--adv-thresh", type=float, default=2_000_000)
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print(f"\n{'='*70}")
    print(f"SHORT-ONLY BACKTEST")
    print(f"{'='*70}")
    print(f"Alpha: {args.alpha_column}")
    print(f"Shorting bottom {args.top_n_short} stocks by alpha")
    print(f"Rebalance every {args.rebalance_every} days")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"ADV threshold: ${args.adv_thresh:,.0f}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading data...")
    df = con.execute(f"""
        SELECT 
            date,
            ticker,
            {args.alpha_column} as alpha,
            {args.target_column} as fwd_ret,
            ret_1d,
            adv_20
        FROM feat_matrix
        WHERE {args.alpha_column} IS NOT NULL 
          AND {args.target_column} IS NOT NULL
          AND ret_1d IS NOT NULL
          AND adv_20 >= {args.adv_thresh}
          AND date >= '{args.start_date}'
          AND date <= '{args.end_date}'
        ORDER BY date, ticker
    """).df()
    
    print(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates")

    # Get unique dates
    dates = sorted(df['date'].unique())
    
    # Simulate short-only portfolio
    print("Running short-only simulation...")
    
    portfolio_returns = []
    rebalance_dates = []
    current_shorts = set()
    
    for i, date in enumerate(dates):
        day_df = df[df['date'] == date].copy()
        
        # Rebalance?
        if i % args.rebalance_every == 0:
            # Select bottom N by alpha (these are short candidates)
            day_df = day_df.sort_values('alpha', ascending=True)
            current_shorts = set(day_df.head(args.top_n_short)['ticker'].tolist())
            rebalance_dates.append(date)
        
        # Calculate daily return for short portfolio
        # Short return = -1 * stock return (we profit when stocks go down)
        short_df = day_df[day_df['ticker'].isin(current_shorts)]
        
        if len(short_df) > 0:
            # Equal weight
            daily_ret = -1 * short_df['ret_1d'].mean()  # Negative because we're short
            portfolio_returns.append({'date': date, 'ret': daily_ret})
    
    # Convert to series
    ret_df = pd.DataFrame(portfolio_returns)
    ret_df['date'] = pd.to_datetime(ret_df['date'])
    ret_df = ret_df.set_index('date')['ret']
    
    # Compute metrics
    metrics = compute_metrics(ret_df)
    
    print(f"\n{'='*70}")
    print(f"SHORT-ONLY RESULTS (bottom {args.top_n_short} stocks)")
    print(f"{'='*70}")
    print(f"Total Return:     {metrics['total_ret']:>10.2f}%")
    print(f"Annual Return:    {metrics['ann_ret']:>10.2f}%")
    print(f"Annual Vol:       {metrics['ann_vol']:>10.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_dd']:>10.2f}%")
    print(f"{'='*70}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if metrics['ann_ret'] > 0:
        print("✓ Short book is PROFITABLE - bottom-alpha stocks underperform")
        print("  L/S weakness may be due to implementation, not signal")
    elif metrics['ann_ret'] > -5:
        print("~ Short book is FLAT - alpha doesn't predict losers well")
        print("  Consider asymmetric L/S (heavier long, lighter short)")
    else:
        print("✗ Short book is LOSING MONEY - bottom-alpha stocks outperform!")
        print("  This explains L/S underperformance. Alpha is one-sided.")
        print("  Options:")
        print("    1. Go long-only")
        print("    2. Use different signal for shorts")
        print("    3. Short high-vol/low-quality instead of low-alpha")

    # Year-by-year breakdown
    print(f"\n\nYEAR-BY-YEAR SHORT BOOK PERFORMANCE:")
    print("-" * 50)
    
    ret_df_yearly = ret_df.copy()
    ret_df_yearly = pd.DataFrame(ret_df_yearly)
    ret_df_yearly['year'] = ret_df_yearly.index.year
    
    for year in sorted(ret_df_yearly['year'].unique()):
        year_rets = ret_df_yearly[ret_df_yearly['year'] == year]['ret']
        year_total = (1 + year_rets).prod() - 1
        print(f"  {year}: {year_total*100:>7.2f}%")

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()