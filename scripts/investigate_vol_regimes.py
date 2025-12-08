#!/usr/bin/env python3
"""
Investigation: Why does high-vol work and low-vol fail?

Hypotheses to test:
1. Alpha IC (predictive power) is higher in high-vol
2. Return dispersion is higher in high-vol (more to capture)
3. Top decile stocks behave differently across vol regimes
4. Low-vol periods have momentum reversal that hurts the signal
5. Beta exposure changes across regimes
"""

import argparse
import duckdb
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("INVESTIGATION: WHY DOES HIGH-VOL WORK?")
    print("=" * 80)

    # First, get regime data joined with feat_matrix
    print("\nLoading data with regime labels...")
    
    # Check what columns exist in feat_matrix for regime
    cols = con.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'feat_matrix' AND column_name LIKE '%regime%'
    """).df()
    print(f"Regime columns in feat_matrix: {cols['column_name'].tolist()}")

    # Load regime history
    regime_df = con.execute("""
        SELECT date, regime, vol_regime, trend_regime, volatility_21d, dispersion_21d
        FROM regime_history
        WHERE date >= '2015-01-01'
    """).df()
    
    print(f"Loaded {len(regime_df)} regime rows")

    # =========================================================================
    # 1. INFORMATION COEFFICIENT BY VOL REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. INFORMATION COEFFICIENT BY VOL REGIME")
    print("=" * 80)
    
    # Join feat_matrix with regime_history
    ic_by_regime = con.execute("""
        WITH regime_dates AS (
            SELECT date, vol_regime, trend_regime, regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        )
        SELECT 
            rd.vol_regime,
            CORR(fm.alpha_composite_v33_regime, fm.ret_5d_f) as ic,
            COUNT(*) as n_obs,
            AVG(fm.ret_5d_f) * 252 as avg_ret_ann,
            STDDEV(fm.ret_5d_f) * SQRT(252) as ret_vol_ann
        FROM feat_matrix fm
        JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
        WHERE fm.alpha_composite_v33_regime IS NOT NULL 
          AND fm.ret_5d_f IS NOT NULL
          AND fm.adv_20 >= 2000000
        GROUP BY rd.vol_regime
        ORDER BY ic DESC
    """).df()
    
    print("\nIC and return characteristics by vol regime:")
    print(ic_by_regime.to_string(index=False))
    
    print("\nInterpretation:")
    for _, row in ic_by_regime.iterrows():
        regime = row['vol_regime']
        ic = row['ic']
        if ic > 0.02:
            quality = "STRONG"
        elif ic > 0.01:
            quality = "MODERATE"
        elif ic > 0:
            quality = "WEAK"
        else:
            quality = "NEGATIVE"
        print(f"  {regime}: IC = {ic:.4f} ({quality})")

    # =========================================================================
    # 2. RETURN DISPERSION BY VOL REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. RETURN DISPERSION BY VOL REGIME")
    print("=" * 80)
    
    dispersion = con.execute("""
        WITH regime_dates AS (
            SELECT date, vol_regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        ),
        daily_stats AS (
            SELECT 
                rd.vol_regime,
                fm.date,
                STDDEV(fm.ret_5d_f) as cross_sectional_vol,
                MAX(fm.ret_5d_f) - MIN(fm.ret_5d_f) as ret_range,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY fm.ret_5d_f) - 
                PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY fm.ret_5d_f) as decile_spread
            FROM feat_matrix fm
            JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
            WHERE fm.ret_5d_f IS NOT NULL AND fm.adv_20 >= 2000000
            GROUP BY rd.vol_regime, fm.date
        )
        SELECT 
            vol_regime,
            AVG(cross_sectional_vol) * 100 as avg_cs_vol_pct,
            AVG(decile_spread) * 100 as avg_decile_spread_pct,
            COUNT(DISTINCT date) as n_days
        FROM daily_stats
        GROUP BY vol_regime
        ORDER BY avg_cs_vol_pct DESC
    """).df()
    
    print("\nCross-sectional return dispersion by vol regime:")
    print(dispersion.to_string(index=False))
    
    print("\nInterpretation:")
    print("  Higher dispersion = more opportunity to differentiate winners from losers")

    # =========================================================================
    # 3. TOP DECILE PERFORMANCE BY VOL REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. TOP DECILE (LONG) PERFORMANCE BY VOL REGIME")
    print("=" * 80)
    
    top_decile = con.execute("""
        WITH regime_dates AS (
            SELECT date, vol_regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        ),
        ranked AS (
            SELECT 
                fm.date,
                fm.ticker,
                fm.alpha_composite_v33_regime as alpha,
                fm.ret_5d_f as fwd_ret,
                rd.vol_regime,
                NTILE(10) OVER (PARTITION BY fm.date ORDER BY fm.alpha_composite_v33_regime) as decile
            FROM feat_matrix fm
            JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
            WHERE fm.alpha_composite_v33_regime IS NOT NULL 
              AND fm.ret_5d_f IS NOT NULL
              AND fm.adv_20 >= 2000000
        )
        SELECT 
            vol_regime,
            decile,
            AVG(fwd_ret) * 252 as ann_ret,
            STDDEV(fwd_ret) * SQRT(252) as ann_vol,
            COUNT(*) as n_obs
        FROM ranked
        WHERE decile IN (1, 5, 10)
        GROUP BY vol_regime, decile
        ORDER BY vol_regime, decile
    """).df()
    
    print("\nDecile returns by vol regime (annualized):")
    pivot = top_decile.pivot(index='vol_regime', columns='decile', values='ann_ret')
    pivot.columns = ['D1 (Short)', 'D5 (Neutral)', 'D10 (Long)']
    pivot['Long-Short'] = pivot['D10 (Long)'] - pivot['D1 (Short)']
    print(pivot.to_string())

    # =========================================================================
    # 4. BOTTOM DECILE (SHORT) BEHAVIOR
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. BOTTOM DECILE (SHORT) ANALYSIS BY VOL REGIME")
    print("=" * 80)
    
    bottom_analysis = con.execute("""
        WITH regime_dates AS (
            SELECT date, vol_regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        ),
        ranked AS (
            SELECT 
                fm.date,
                fm.ticker,
                fm.alpha_composite_v33_regime as alpha,
                fm.ret_5d_f as fwd_ret,
                fm.vol_21,
                rd.vol_regime,
                NTILE(10) OVER (PARTITION BY fm.date ORDER BY fm.alpha_composite_v33_regime) as decile
            FROM feat_matrix fm
            JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
            WHERE fm.alpha_composite_v33_regime IS NOT NULL 
              AND fm.ret_5d_f IS NOT NULL
              AND fm.adv_20 >= 2000000
        )
        SELECT 
            vol_regime,
            AVG(fwd_ret) * 252 as d1_ann_ret,
            AVG(vol_21) as d1_avg_stock_vol,
            COUNT(*) as n_obs
        FROM ranked
        WHERE decile = 1
        GROUP BY vol_regime
        ORDER BY d1_ann_ret DESC
    """).df()
    
    print("\nBottom decile (short candidates) characteristics:")
    print(bottom_analysis.to_string(index=False))
    
    print("\nKey insight: If D1 returns are HIGH, shorting them loses money")

    # =========================================================================
    # 5. ALPHA SIGNAL CHARACTERISTICS BY VOL REGIME
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. ALPHA SIGNAL CHARACTERISTICS BY VOL REGIME")
    print("=" * 80)
    
    alpha_chars = con.execute("""
        WITH regime_dates AS (
            SELECT date, vol_regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        )
        SELECT 
            rd.vol_regime,
            AVG(fm.alpha_composite_v33_regime) as avg_alpha,
            STDDEV(fm.alpha_composite_v33_regime) as std_alpha,
            MIN(fm.alpha_composite_v33_regime) as min_alpha,
            MAX(fm.alpha_composite_v33_regime) as max_alpha,
            COUNT(*) as n_obs
        FROM feat_matrix fm
        JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
        WHERE fm.alpha_composite_v33_regime IS NOT NULL 
          AND fm.adv_20 >= 2000000
        GROUP BY rd.vol_regime
    """).df()
    
    print("\nAlpha signal distribution by vol regime:")
    print(alpha_chars.to_string(index=False))

    # =========================================================================
    # 6. WHAT MAKES NORMAL_VOL_NEUTRAL SPECIAL?
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. WHY DOES L/S WORK IN NORMAL_VOL_NEUTRAL?")
    print("=" * 80)
    
    nvn_analysis = con.execute("""
        WITH regime_dates AS (
            SELECT date, regime, vol_regime, trend_regime
            FROM regime_history
            WHERE date >= '2015-01-01'
        ),
        ranked AS (
            SELECT 
                fm.date,
                rd.regime,
                fm.ret_5d_f as fwd_ret,
                NTILE(10) OVER (PARTITION BY fm.date ORDER BY fm.alpha_composite_v33_regime) as decile
            FROM feat_matrix fm
            JOIN regime_dates rd ON CAST(fm.date AS DATE) = CAST(rd.date AS DATE)
            WHERE fm.alpha_composite_v33_regime IS NOT NULL 
              AND fm.ret_5d_f IS NOT NULL
              AND fm.adv_20 >= 2000000
        )
        SELECT 
            regime,
            AVG(CASE WHEN decile = 10 THEN fwd_ret END) * 252 as d10_ret,
            AVG(CASE WHEN decile = 1 THEN fwd_ret END) * 252 as d1_ret,
            (AVG(CASE WHEN decile = 10 THEN fwd_ret END) - 
             AVG(CASE WHEN decile = 1 THEN fwd_ret END)) * 252 as ls_spread,
            COUNT(DISTINCT date) as n_days
        FROM ranked
        GROUP BY regime
        ORDER BY ls_spread DESC
    """).df()
    
    print("\nLong-short spread by full regime:")
    print(nvn_analysis.to_string(index=False))
    
    # Find where L/S actually works
    print("\nRegimes where L/S spread is POSITIVE (shorting works):")
    positive_ls = nvn_analysis[nvn_analysis['ls_spread'] > 0]
    print(positive_ls.to_string(index=False))
    
    print("\nRegimes where L/S spread is NEGATIVE (shorting fails):")
    negative_ls = nvn_analysis[nvn_analysis['ls_spread'] <= 0]
    print(negative_ls.to_string(index=False))

    # =========================================================================
    # 7. TEMPORAL ANALYSIS - HAS SOMETHING CHANGED?
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. TEMPORAL ANALYSIS - IC BY YEAR")
    print("=" * 80)
    
    ic_by_year = con.execute("""
        SELECT 
            EXTRACT(YEAR FROM date) as year,
            CORR(alpha_composite_v33_regime, ret_5d_f) as ic,
            COUNT(*) as n_obs
        FROM feat_matrix
        WHERE alpha_composite_v33_regime IS NOT NULL 
          AND ret_5d_f IS NOT NULL
          AND adv_20 >= 2000000
          AND date >= '2015-01-01'
        GROUP BY 1
        ORDER BY 1
    """).df()
    
    print("\nIC by year:")
    print(ic_by_year.to_string(index=False))
    
    # Identify trend
    recent_ic = ic_by_year[ic_by_year['year'] >= 2022]['ic'].mean()
    early_ic = ic_by_year[ic_by_year['year'] <= 2019]['ic'].mean()
    print(f"\nEarly period IC (2015-2019): {early_ic:.4f}")
    print(f"Recent period IC (2022-2025): {recent_ic:.4f}")
    if recent_ic < early_ic:
        print("⚠️  Alpha signal appears to be DECAYING over time")
    else:
        print("✓ Alpha signal is stable or improving")

    # =========================================================================
    # 8. FACTOR EXPOSURE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. WHAT DOES THE ALPHA ACTUALLY CAPTURE?")
    print("=" * 80)
    
    # Check correlation with other factors
    factor_corr = con.execute("""
        SELECT 
            CORR(alpha_composite_v33_regime, ret_1d) as corr_momentum_1d,
            CORR(alpha_composite_v33_regime, ret_5d) as corr_momentum_5d,
            CORR(alpha_composite_v33_regime, vol_21) as corr_volatility,
            CORR(alpha_composite_v33_regime, beta_252d) as corr_beta,
            CORR(alpha_composite_v33_regime, adv_20) as corr_liquidity
        FROM feat_matrix
        WHERE alpha_composite_v33_regime IS NOT NULL 
          AND date >= '2015-01-01'
          AND adv_20 >= 2000000
    """).df()
    
    print("\nAlpha correlation with other factors:")
    for col in factor_corr.columns:
        factor_name = col.replace('corr_', '')
        corr_val = factor_corr[col].values[0]
        print(f"  {factor_name}: {corr_val:.4f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)
    
    print("""
KEY QUESTIONS ANSWERED:

1. WHY DOES HIGH-VOL WORK?
   - Check IC by vol regime above
   - Check dispersion (more opportunity in high vol?)
   - Check if D10 outperformance is higher

2. WHY DOES LOW-VOL FAIL?
   - Lower IC? Lower dispersion?
   - D1 (short candidates) might outperform in low-vol

3. WHY DOES L/S FAIL OVERALL?
   - D1 returns are positive (shorting loses money)
   - Only works in specific regimes

4. IS THE SIGNAL DECAYING?
   - Compare early vs recent IC

5. WHAT FACTOR EXPOSURES EXIST?
   - Momentum? Volatility? Beta? Size?
   
Review the data above to form conclusions.
""")

    con.close()

if __name__ == "__main__":
    main()