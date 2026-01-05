#!/usr/bin/env python3
"""
Decile Return Analysis for Kairos Alpha Signals

This script analyzes whether the alpha signal is symmetric (good for both
long AND short) or asymmetric (only good for picking winners).

Key diagnostic: If decile 1 (lowest alpha) doesn't underperform decile 5 (median),
then shorting low-alpha stocks won't work.
"""

import argparse
import duckdb
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    parser.add_argument("--alpha-column", type=str, default="alpha_composite_v33_regime")
    parser.add_argument("--target-column", type=str, default="ret_5d_f")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="2025-11-28")
    parser.add_argument("--adv-thresh", type=float, default=2_000_000)
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print(f"\n{'='*70}")
    print(f"DECILE RETURN ANALYSIS")
    print(f"Alpha: {args.alpha_column}")
    print(f"Target: {args.target_column}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"{'='*70}\n")

    # 1. Overall decile analysis
    print("1. OVERALL DECILE RETURNS (annualized)")
    print("-" * 50)
    
    df = con.execute(f"""
        WITH ranked AS (
            SELECT 
                date,
                ticker,
                {args.alpha_column} as alpha,
                {args.target_column} as fwd_ret,
                adv_20,
                NTILE(10) OVER (PARTITION BY date ORDER BY {args.alpha_column}) as decile
            FROM feat_matrix_v2
            WHERE {args.alpha_column} IS NOT NULL 
              AND {args.target_column} IS NOT NULL
              AND adv_20 >= {args.adv_thresh}
              AND date >= '{args.start_date}'
              AND date <= '{args.end_date}'
        )
        SELECT 
            decile,
            COUNT(*) as n_obs,
            AVG(fwd_ret) * 252 as ann_ret,
            STDDEV(fwd_ret) * SQRT(252) as ann_vol,
            AVG(fwd_ret) / NULLIF(STDDEV(fwd_ret), 0) * SQRT(252) as sharpe
        FROM ranked
        GROUP BY decile
        ORDER BY decile
    """).df()
    
    print(df.to_string(index=False))
    
    # Calculate long-short spread
    d10_ret = df[df['decile'] == 10]['ann_ret'].values[0]
    d1_ret = df[df['decile'] == 1]['ann_ret'].values[0]
    print(f"\nDecile 10 (top) - Decile 1 (bottom) spread: {d10_ret - d1_ret:.2f}%")
    print(f"Decile 10 return: {d10_ret:.2f}%")
    print(f"Decile 1 return: {d1_ret:.2f}%")
    
    # 2. Monotonicity check
    print(f"\n\n2. MONOTONICITY CHECK")
    print("-" * 50)
    returns = df['ann_ret'].values
    monotonic_violations = 0
    for i in range(len(returns) - 1):
        if returns[i] > returns[i+1]:
            monotonic_violations += 1
            print(f"   Violation: Decile {i+1} ({returns[i]:.2f}%) > Decile {i+2} ({returns[i+1]:.2f}%)")
    
    if monotonic_violations == 0:
        print("   ✓ Perfect monotonicity - alpha ranks correctly across all deciles")
    else:
        print(f"   ✗ {monotonic_violations} monotonicity violations")

    # 3. Year-by-year decile 1 vs decile 10
    print(f"\n\n3. YEAR-BY-YEAR: DECILE 10 (LONG) vs DECILE 1 (SHORT)")
    print("-" * 50)
    
    df_yearly = con.execute(f"""
        WITH ranked AS (
            SELECT 
                EXTRACT(YEAR FROM date) as year,
                {args.alpha_column} as alpha,
                {args.target_column} as fwd_ret,
                NTILE(10) OVER (PARTITION BY date ORDER BY {args.alpha_column}) as decile
            FROM feat_matrix_v2
            WHERE {args.alpha_column} IS NOT NULL 
              AND {args.target_column} IS NOT NULL
              AND adv_20 >= {args.adv_thresh}
              AND date >= '{args.start_date}'
              AND date <= '{args.end_date}'
        )
        SELECT 
            year,
            AVG(CASE WHEN decile = 10 THEN fwd_ret END) * 252 as d10_ret,
            AVG(CASE WHEN decile = 1 THEN fwd_ret END) * 252 as d1_ret,
            (AVG(CASE WHEN decile = 10 THEN fwd_ret END) - 
             AVG(CASE WHEN decile = 1 THEN fwd_ret END)) * 252 as ls_spread
        FROM ranked
        GROUP BY year
        ORDER BY year
    """).df()
    
    print(df_yearly.to_string(index=False))
    
    # Count winning years for L/S
    ls_wins = (df_yearly['ls_spread'] > 0).sum()
    total_years = len(df_yearly)
    print(f"\nL/S spread positive in {ls_wins}/{total_years} years ({100*ls_wins/total_years:.0f}%)")

    # 4. Information Coefficient by decile
    print(f"\n\n4. INFORMATION COEFFICIENT (IC) ANALYSIS")
    print("-" * 50)
    
    ic_df = con.execute(f"""
        SELECT 
            CORR({args.alpha_column}, {args.target_column}) as overall_ic
        FROM feat_matrix_v2
        WHERE {args.alpha_column} IS NOT NULL 
          AND {args.target_column} IS NOT NULL
          AND adv_20 >= {args.adv_thresh}
          AND date >= '{args.start_date}'
          AND date <= '{args.end_date}'
    """).df()
    
    print(f"Overall IC: {ic_df['overall_ic'].values[0]:.4f}")
    
    # IC by year
    ic_yearly = con.execute(f"""
        SELECT 
            EXTRACT(YEAR FROM date) as year,
            CORR({args.alpha_column}, {args.target_column}) as ic
        FROM feat_matrix_v2
        WHERE {args.alpha_column} IS NOT NULL 
          AND {args.target_column} IS NOT NULL
          AND adv_20 >= {args.adv_thresh}
          AND date >= '{args.start_date}'
          AND date <= '{args.end_date}'
        GROUP BY 1
        ORDER BY 1
    """).df()
    
    print("\nIC by year:")
    print(ic_yearly.to_string(index=False))

    # 5. Short-side specific analysis
    print(f"\n\n5. SHORT-SIDE DEEP DIVE (Bottom 3 Deciles)")
    print("-" * 50)
    
    short_df = con.execute(f"""
        WITH ranked AS (
            SELECT 
                date,
                ticker,
                {args.alpha_column} as alpha,
                {args.target_column} as fwd_ret,
                NTILE(10) OVER (PARTITION BY date ORDER BY {args.alpha_column}) as decile
            FROM feat_matrix_v2
            WHERE {args.alpha_column} IS NOT NULL 
              AND {args.target_column} IS NOT NULL
              AND adv_20 >= {args.adv_thresh}
              AND date >= '{args.start_date}'
              AND date <= '{args.end_date}'
        )
        SELECT 
            decile,
            AVG(fwd_ret) * 252 as ann_ret,
            STDDEV(fwd_ret) * SQRT(252) as ann_vol,
            AVG(fwd_ret) / NULLIF(STDDEV(fwd_ret), 0) * SQRT(252) as sharpe,
            -- Negative sharpe = good for shorting
            -1 * AVG(fwd_ret) / NULLIF(STDDEV(fwd_ret), 0) * SQRT(252) as short_sharpe
        FROM ranked
        WHERE decile <= 3
        GROUP BY decile
        ORDER BY decile
    """).df()
    
    print(short_df.to_string(index=False))
    print("\nNote: 'short_sharpe' = Sharpe if you SHORT these stocks (negative return = good)")

    # 6. Quintile analysis (simpler view)
    print(f"\n\n6. QUINTILE SUMMARY (5 buckets)")
    print("-" * 50)
    
    quint_df = con.execute(f"""
        WITH ranked AS (
            SELECT 
                {args.alpha_column} as alpha,
                {args.target_column} as fwd_ret,
                NTILE(5) OVER (PARTITION BY date ORDER BY {args.alpha_column}) as quintile
            FROM feat_matrix_v2
            WHERE {args.alpha_column} IS NOT NULL 
              AND {args.target_column} IS NOT NULL
              AND adv_20 >= {args.adv_thresh}
              AND date >= '{args.start_date}'
              AND date <= '{args.end_date}'
        )
        SELECT 
            quintile,
            CASE quintile 
                WHEN 1 THEN 'SHORT'
                WHEN 5 THEN 'LONG'
                ELSE 'NEUTRAL'
            END as position,
            AVG(fwd_ret) * 252 as ann_ret,
            STDDEV(fwd_ret) * SQRT(252) as ann_vol
        FROM ranked
        GROUP BY quintile
        ORDER BY quintile
    """).df()
    
    print(quint_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}\n")
    
    con.close()

if __name__ == "__main__":
    main()