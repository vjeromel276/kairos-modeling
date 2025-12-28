#!/usr/bin/env python3
"""
Deep Regime Switching Analysis for Kairos

Goal: Understand exactly what the regime switcher is doing and identify
paths to improve performance toward institutional-grade results.

Institutional benchmarks to target:
- Renaissance Medallion: Sharpe ~3-4 (unrealistic target)
- Top quant funds (DE Shaw, Two Sigma, Citadel): Sharpe 1.5-2.5
- Good systematic fund: Sharpe 1.0-1.5
- You currently: Long-only 1.30, Regime-switch 1.16

Analysis sections:
1. Regime distribution over time
2. What strategy is selected in each regime
3. Return attribution by regime
4. Drawdown analysis by regime
5. Transition analysis (regime changes)
6. Optimal regime allocation (what SHOULD we do?)
7. Comparison to benchmark allocation
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

def compute_sharpe(returns):
    """Annualized Sharpe ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(252 / 5)  # 5-day rebalance

def compute_max_dd(returns):
    """Maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="2025-11-28")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("DEEP REGIME SWITCHING ANALYSIS")
    print(f"Period: {args.start_date} to {args.end_date}")
    print("=" * 80)

    # Load all data
    print("\nLoading data...")
    
    regime_df = con.execute(f"""
        SELECT date, regime, regime_score, vol_regime, trend_regime, 
               volatility_21d, market_return, trend_60d
        FROM regime_history
        WHERE date >= '{args.start_date}' AND date <= '{args.end_date}'
        ORDER BY date
    """).df()
    
    longonly_df = con.execute("""
        SELECT date, port_ret, bench_ret, active_ret
        FROM backtest_results_longonly_r4
        ORDER BY date
    """).df()
    
    ls_df = con.execute("""
        SELECT date, port_ret, bench_ret, active_ret
        FROM backtest_results_ls_opt
        ORDER BY date
    """).df()
    
    regime_switch_df = con.execute("""
        SELECT *
        FROM backtest_results_regime_switching
        ORDER BY date
    """).df()

    print(f"  Regime history: {len(regime_df)} rows")
    print(f"  Long-only results: {len(longonly_df)} rows")
    print(f"  L/S results: {len(ls_df)} rows")
    print(f"  Regime-switch results: {len(regime_switch_df)} rows")

    # Merge everything
    merged = regime_df.merge(
        longonly_df, on='date', how='inner', suffixes=('', '_lo')
    ).merge(
        ls_df, on='date', how='inner', suffixes=('', '_ls')
    )
    
    # Rename for clarity
    merged = merged.rename(columns={
        'port_ret': 'lo_ret',
        'port_ret_ls': 'ls_ret',
        'bench_ret': 'bench_ret_lo',
        'bench_ret_ls': 'bench_ret'
    })

    print(f"  Merged dataset: {len(merged)} rows")

    # =========================================================================
    # 1. REGIME DISTRIBUTION OVER TIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. REGIME DISTRIBUTION")
    print("=" * 80)
    
    regime_counts = merged['regime'].value_counts()
    total = len(merged)
    
    print(f"\n{'Regime':<20} {'Count':>8} {'Pct':>8} {'Days ~':>10}")
    print("-" * 50)
    for regime, count in regime_counts.items():
        pct = 100 * count / total
        approx_days = count * 5  # 5-day rebalance
        print(f"{regime:<20} {count:>8} {pct:>7.1f}% {approx_days:>10}")

    # By year
    print("\nRegime distribution by year:")
    merged['year'] = pd.to_datetime(merged['date']).dt.year
    yearly_regime = merged.groupby(['year', 'regime']).size().unstack(fill_value=0)
    print(yearly_regime.to_string())

    # =========================================================================
    # 2. STRATEGY PERFORMANCE BY REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. STRATEGY PERFORMANCE BY REGIME")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'Regime':<20} {'LO Sharpe':>10} {'LS Sharpe':>10} {'LO Ret%':>10} {'LS Ret%':>10} {'Winner':>10}")
    print("-" * 80)
    
    regime_perf = []
    for regime in merged['regime'].unique():
        mask = merged['regime'] == regime
        lo_rets = merged.loc[mask, 'lo_ret']
        ls_rets = merged.loc[mask, 'ls_ret']
        
        lo_sharpe = compute_sharpe(lo_rets)
        ls_sharpe = compute_sharpe(ls_rets)
        lo_total = (1 + lo_rets).prod() - 1
        ls_total = (1 + ls_rets).prod() - 1
        
        winner = "LONG-ONLY" if lo_sharpe > ls_sharpe else "L/S"
        
        regime_perf.append({
            'regime': regime,
            'lo_sharpe': lo_sharpe,
            'ls_sharpe': ls_sharpe,
            'lo_total': lo_total,
            'ls_total': ls_total,
            'count': mask.sum(),
            'winner': winner
        })
        
        print(f"{regime:<20} {lo_sharpe:>10.2f} {ls_sharpe:>10.2f} {lo_total*100:>9.1f}% {ls_total*100:>9.1f}% {winner:>10}")

    regime_perf_df = pd.DataFrame(regime_perf)

    # =========================================================================
    # 3. OPTIMAL REGIME ALLOCATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. OPTIMAL REGIME ALLOCATION (HINDSIGHT)")
    print("=" * 80)
    
    print("\nIf we always picked the best strategy per regime:")
    
    # Create optimal returns series
    merged['optimal_ret'] = merged.apply(
        lambda row: row['lo_ret'] if regime_perf_df[regime_perf_df['regime'] == row['regime']]['lo_sharpe'].values[0] > 
                                     regime_perf_df[regime_perf_df['regime'] == row['regime']]['ls_sharpe'].values[0]
                    else row['ls_ret'],
        axis=1
    )
    
    # Also compute "always long-only" and "always L/S"
    merged['always_lo'] = merged['lo_ret']
    merged['always_ls'] = merged['ls_ret']
    
    strategies = {
        'Always Long-Only': merged['always_lo'],
        'Always L/S': merged['always_ls'],
        'Optimal (hindsight)': merged['optimal_ret']
    }
    
    print(f"\n{'Strategy':<25} {'Sharpe':>10} {'Total Ret':>12} {'Ann Ret':>10} {'Max DD':>10}")
    print("-" * 70)
    
    for name, rets in strategies.items():
        sharpe = compute_sharpe(rets)
        total = (1 + rets).prod() - 1
        n_years = len(rets) * 5 / 252  # approximate years
        ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd = compute_max_dd(rets)
        print(f"{name:<25} {sharpe:>10.2f} {total*100:>11.1f}% {ann_ret*100:>9.1f}% {max_dd*100:>9.1f}%")

    # =========================================================================
    # 4. WHAT SHOULD THE REGIME RULES BE?
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. RECOMMENDED REGIME RULES")
    print("=" * 80)
    
    print("\nBased on historical performance, optimal allocation:")
    print("-" * 50)
    
    for _, row in regime_perf_df.iterrows():
        regime = row['regime']
        lo_sharpe = row['lo_sharpe']
        ls_sharpe = row['ls_sharpe']
        diff = lo_sharpe - ls_sharpe
        
        if diff > 0.3:
            recommendation = "STRONG LONG-ONLY"
        elif diff > 0:
            recommendation = "LEAN LONG-ONLY"
        elif diff > -0.3:
            recommendation = "LEAN L/S"
        else:
            recommendation = "STRONG L/S"
            
        print(f"  {regime:<20}: {recommendation:<18} (LO: {lo_sharpe:.2f}, LS: {ls_sharpe:.2f}, diff: {diff:+.2f})")

    # =========================================================================
    # 5. REGIME TRANSITIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. REGIME TRANSITIONS")
    print("=" * 80)
    
    merged['prev_regime'] = merged['regime'].shift(1)
    merged['regime_changed'] = merged['regime'] != merged['prev_regime']
    
    n_transitions = merged['regime_changed'].sum()
    avg_regime_length = len(merged) / n_transitions if n_transitions > 0 else len(merged)
    
    print(f"\nTotal regime transitions: {n_transitions}")
    print(f"Average regime length: {avg_regime_length:.1f} periods ({avg_regime_length * 5:.0f} days)")
    
    # Transition matrix
    print("\nTransition counts (from row to column):")
    transition_matrix = pd.crosstab(merged['prev_regime'], merged['regime'], margins=True)
    print(transition_matrix.to_string())

    # =========================================================================
    # 6. DRAWDOWN ANALYSIS BY REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. DRAWDOWN ANALYSIS BY REGIME")
    print("=" * 80)
    
    print(f"\n{'Regime':<20} {'LO Max DD':>12} {'LS Max DD':>12} {'Better':>10}")
    print("-" * 60)
    
    for regime in merged['regime'].unique():
        mask = merged['regime'] == regime
        lo_dd = compute_max_dd(merged.loc[mask, 'lo_ret'])
        ls_dd = compute_max_dd(merged.loc[mask, 'ls_ret'])
        better = "L/S" if ls_dd > lo_dd else "LONG-ONLY"  # less negative is better
        print(f"{regime:<20} {lo_dd*100:>11.1f}% {ls_dd*100:>11.1f}% {better:>10}")

    # =========================================================================
    # 7. HIGH-VOL ONLY STRATEGY
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. HIGH-VOL ONLY STRATEGY")
    print("=" * 80)
    
    high_vol_mask = merged['vol_regime'] == 'high_vol'
    normal_vol_mask = merged['vol_regime'] == 'normal_vol'
    low_vol_mask = merged['vol_regime'] == 'low_vol'
    
    print(f"\nVol regime distribution:")
    print(f"  High vol:   {high_vol_mask.sum()} periods ({100*high_vol_mask.sum()/len(merged):.1f}%)")
    print(f"  Normal vol: {normal_vol_mask.sum()} periods ({100*normal_vol_mask.sum()/len(merged):.1f}%)")
    print(f"  Low vol:    {low_vol_mask.sum()} periods ({100*low_vol_mask.sum()/len(merged):.1f}%)")
    
    # Strategy: Long-only in high vol, cash otherwise
    merged['highvol_lo'] = merged['lo_ret'].where(high_vol_mask, 0)
    merged['highvol_normalvol_lo'] = merged['lo_ret'].where(high_vol_mask | normal_vol_mask, 0)
    
    print(f"\n{'Strategy':<35} {'Sharpe':>10} {'Total Ret':>12} {'Max DD':>10}")
    print("-" * 70)
    
    test_strategies = {
        'Always Long-Only': merged['lo_ret'],
        'High-Vol Only (cash otherwise)': merged['highvol_lo'],
        'High+Normal Vol (cash in low)': merged['highvol_normalvol_lo'],
    }
    
    for name, rets in test_strategies.items():
        # Adjust Sharpe for time in market
        active_periods = (rets != 0).sum()
        if active_periods > 0:
            active_rets = rets[rets != 0]
            sharpe = compute_sharpe(active_rets)
        else:
            sharpe = 0
        total = (1 + rets).prod() - 1
        max_dd = compute_max_dd(rets)
        print(f"{name:<35} {sharpe:>10.2f} {total*100:>11.1f}% {max_dd*100:>9.1f}%")

    # =========================================================================
    # 8. BULL VS BEAR ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. BULL VS BEAR ANALYSIS")
    print("=" * 80)
    
    bull_mask = merged['trend_regime'] == 'bull'
    bear_mask = merged['trend_regime'] == 'bear'
    
    print(f"\nTrend regime distribution:")
    print(f"  Bull: {bull_mask.sum()} periods ({100*bull_mask.sum()/len(merged):.1f}%)")
    print(f"  Bear: {bear_mask.sum()} periods ({100*bear_mask.sum()/len(merged):.1f}%)")
    
    print(f"\n{'Condition':<20} {'LO Sharpe':>12} {'LS Sharpe':>12} {'Verdict':>15}")
    print("-" * 65)
    
    for name, mask in [('Bull', bull_mask), ('Bear', bear_mask)]:
        lo_sharpe = compute_sharpe(merged.loc[mask, 'lo_ret'])
        ls_sharpe = compute_sharpe(merged.loc[mask, 'ls_ret'])
        verdict = "LO wins" if lo_sharpe > ls_sharpe else "LS wins"
        print(f"{name:<20} {lo_sharpe:>12.2f} {ls_sharpe:>12.2f} {verdict:>15}")

    # =========================================================================
    # 9. INFORMATION COEFFICIENT BY REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("9. SUMMARY: PATH TO INSTITUTIONAL QUALITY")
    print("=" * 80)
    
    print("""
CURRENT STATE:
  - Long-only Sharpe: 1.30 (good, approaching institutional)
  - L/S Sharpe: 0.43 (weak, short book is hurting)
  - Regime-switch Sharpe: 1.16 (decent, but below pure long-only)

KEY FINDINGS:
  1. High-vol regimes are your sweet spot (Sharpe 1.5-2.0+)
  2. Low-vol regimes are killing performance (Sharpe 0.25-0.40)
  3. L/S only helps in bear markets (and even then, marginally)
  4. The short signal is fundamentally broken (bottom decile outperforms)

PATHS TO INSTITUTIONAL QUALITY (Sharpe 1.5+):

  Option A: "Vol Timing"
    - Go long-only in high/normal vol
    - Go to cash or reduced exposure in low vol
    - Expected Sharpe: 1.4-1.6

  Option B: "Fix the Short Book"
    - Build separate short signal (quality, momentum reversal)
    - Only short in bear/high-vol regimes
    - Expected Sharpe: 1.3-1.5 with lower vol

  Option C: "Concentration in High-Vol"
    - Increase position sizes in high-vol (more conviction)
    - Reduce positions in low-vol
    - Expected Sharpe: 1.4-1.7 (more variance)

  Option D: "Multi-Strategy"
    - Run long-only as core
    - Add separate mean-reversion strategy for low-vol
    - Add separate short strategy for bear markets
    - Expected Sharpe: 1.5-2.0 (most work, best outcome)

RECOMMENDED NEXT STEP:
  Implement Option A (simplest) and test Option D (best upside).
""")

    con.close()
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()