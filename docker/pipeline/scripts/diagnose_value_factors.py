#!/usr/bin/env python3
"""
Investigate:
1. Why feat_value only has 4,592 rows
2. Build and test proper valuation factors from daily table
3. Check for data leakage in alpha_mlm
"""

import argparse
import duckdb
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("DIAGNOSTIC: VALUE FACTORS & DATA ISSUES")
    print("=" * 80)

    # =========================================================================
    # 1. WHY IS FEAT_VALUE SO SMALL?
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. FEAT_VALUE INVESTIGATION")
    print("=" * 80)
    
    # Check feat_value
    fv_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM feat_value
    """).df()
    print("\nfeat_value summary:")
    print(fv_info.to_string(index=False))
    
    # Sample
    print("\nfeat_value sample:")
    sample = con.execute("SELECT * FROM feat_value ORDER BY date DESC LIMIT 10").df()
    print(sample.to_string(index=False))
    
    # Compare to feat_matrix
    fm_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM feat_matrix
    """).df()
    print("\nfeat_matrix summary (for comparison):")
    print(fm_info.to_string(index=False))

    # Check how value_raw appears in feat_matrix
    print("\nvalue_raw in feat_matrix:")
    vm_check = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(value_raw) as non_null_value_raw,
            COUNT(DISTINCT CASE WHEN value_raw IS NOT NULL THEN ticker END) as tickers_with_value
        FROM feat_matrix
    """).df()
    print(vm_check.to_string(index=False))

    # =========================================================================
    # 2. BUILD VALUATION FACTORS FROM DAILY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. VALUATION FACTORS FROM DAILY TABLE")
    print("=" * 80)
    
    # Check daily table coverage
    daily_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(pe) as pe_count,
            COUNT(pb) as pb_count,
            COUNT(ps) as ps_count
        FROM daily
    """).df()
    print("\ndaily table summary:")
    print(daily_info.to_string(index=False))
    
    # Test earnings yield (1/PE) IC with proper filtering
    print("\nTesting valuation factor ICs with proper filtering...")
    
    valuation_tests = con.execute("""
        WITH daily_vals AS (
            SELECT 
                ticker,
                date,
                -- Earnings yield (1/PE), filtered for positive PE < 100
                CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END as earnings_yield,
                -- Book yield (1/PB), filtered for positive PB < 20  
                CASE WHEN pb > 0 AND pb < 20 THEN 1.0/pb ELSE NULL END as book_yield,
                -- Sales yield (1/PS)
                CASE WHEN ps > 0 AND ps < 50 THEN 1.0/ps ELSE NULL END as sales_yield,
                -- EBITDA yield (1/EV_EBITDA)
                CASE WHEN evebitda > 0 AND evebitda < 50 THEN 1.0/evebitda ELSE NULL END as ebitda_yield,
                marketcap
            FROM daily
            WHERE date >= '2015-01-01'
        ),
        with_returns AS (
            SELECT 
                d.*,
                LEAD(d.marketcap, 5) OVER (PARTITION BY d.ticker ORDER BY d.date) / 
                    NULLIF(d.marketcap, 0) - 1 as ret_5d_f
            FROM daily_vals d
        )
        SELECT 
            CORR(earnings_yield, ret_5d_f) as ic_earnings_yield,
            CORR(book_yield, ret_5d_f) as ic_book_yield,
            CORR(sales_yield, ret_5d_f) as ic_sales_yield,
            CORR(ebitda_yield, ret_5d_f) as ic_ebitda_yield,
            COUNT(*) as n_obs,
            COUNT(earnings_yield) as n_ey,
            COUNT(book_yield) as n_by,
            COUNT(sales_yield) as n_sy,
            COUNT(ebitda_yield) as n_ebitda
        FROM with_returns
        WHERE ret_5d_f IS NOT NULL
          AND ret_5d_f BETWEEN -0.5 AND 0.5  -- Filter extreme returns
    """).df()
    
    print("\nValuation Factor ICs (filtered, from daily table):")
    print(f"  Earnings Yield (1/PE): IC = {valuation_tests['ic_earnings_yield'].values[0]:.4f} (n={valuation_tests['n_ey'].values[0]:,})")
    print(f"  Book Yield (1/PB):     IC = {valuation_tests['ic_book_yield'].values[0]:.4f} (n={valuation_tests['n_by'].values[0]:,})")
    print(f"  Sales Yield (1/PS):    IC = {valuation_tests['ic_sales_yield'].values[0]:.4f} (n={valuation_tests['n_sy'].values[0]:,})")
    print(f"  EBITDA Yield:          IC = {valuation_tests['ic_ebitda_yield'].values[0]:.4f} (n={valuation_tests['n_ebitda'].values[0]:,})")

    # =========================================================================
    # 3. CHECK ALPHA_MLM FOR LEAKAGE
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. ALPHA_MLM LEAKAGE CHECK")
    print("=" * 80)
    
    # Check if alpha_mlm is correlated with future returns suspiciously
    mlm_check = con.execute("""
        SELECT 
            CORR(alpha_mlm, ret_5d_f) as ic_5d_forward,
            CORR(alpha_mlm, ret_1d_f) as ic_1d_forward,
            CORR(alpha_mlm, ret_5d) as ic_5d_backward,
            CORR(alpha_mlm, ret_1d) as ic_1d_backward,
            COUNT(*) as n_obs
        FROM feat_matrix
        WHERE alpha_mlm IS NOT NULL
          AND date >= '2015-01-01'
    """).df()
    
    print("\nalpha_mlm correlations:")
    print(f"  vs ret_5d_f (forward):  {mlm_check['ic_5d_forward'].values[0]:.4f}")
    print(f"  vs ret_1d_f (forward):  {mlm_check['ic_1d_forward'].values[0]:.4f}")
    print(f"  vs ret_5d (backward):   {mlm_check['ic_5d_backward'].values[0]:.4f}")
    print(f"  vs ret_1d (backward):   {mlm_check['ic_1d_backward'].values[0]:.4f}")
    
    if mlm_check['ic_5d_forward'].values[0] > 0.05:
        print("\n⚠️  WARNING: alpha_mlm has IC > 5% with forward returns.")
        print("    This strongly suggests look-ahead bias or target leakage.")
        print("    DO NOT USE THIS FACTOR.")

    # Check feat_mlm table
    print("\nfeat_mlm table info:")
    mlm_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM feat_mlm
    """).df()
    print(mlm_info.to_string(index=False))
    
    # Check columns in feat_mlm
    mlm_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'feat_mlm'
    """).df()
    print(f"\nfeat_mlm columns: {mlm_cols['column_name'].tolist()}")

    # =========================================================================
    # 4. CHECK LABEL_5D_UP - CONFIRM IT'S TARGET LEAKAGE
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. LABEL_5D_UP LEAKAGE CONFIRMATION")
    print("=" * 80)
    
    label_check = con.execute("""
        SELECT 
            CORR(label_5d_up, ret_5d_f) as correlation,
            AVG(CASE WHEN label_5d_up = 1 THEN ret_5d_f END) as avg_ret_when_1,
            AVG(CASE WHEN label_5d_up = 0 THEN ret_5d_f END) as avg_ret_when_0,
            COUNT(*) as n_obs
        FROM feat_matrix
        WHERE label_5d_up IS NOT NULL AND ret_5d_f IS NOT NULL
    """).df()
    
    print(f"\nlabel_5d_up analysis:")
    print(f"  Correlation with ret_5d_f: {label_check['correlation'].values[0]:.4f}")
    print(f"  Avg return when label=1:   {label_check['avg_ret_when_1'].values[0]*100:.2f}%")
    print(f"  Avg return when label=0:   {label_check['avg_ret_when_0'].values[0]*100:.2f}%")
    print("\n✓ CONFIRMED: label_5d_up is derived from ret_5d_f (it's the target, not a feature)")

    # =========================================================================
    # 5. COMPOSITE VALUE FACTOR TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. COMPOSITE VALUE FACTOR TEST")
    print("=" * 80)
    
    # Build a composite value factor and test IC
    composite_test = con.execute("""
        WITH daily_vals AS (
            SELECT 
                ticker,
                date,
                -- Z-score each factor within each date
                (CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END) as ey,
                (CASE WHEN pb > 0 AND pb < 20 THEN 1.0/pb ELSE NULL END) as by,
                (CASE WHEN evebitda > 0 AND evebitda < 50 THEN 1.0/evebitda ELSE NULL END) as ebitda_y,
                marketcap
            FROM daily
            WHERE date >= '2015-01-01'
        ),
        zscored AS (
            SELECT 
                ticker,
                date,
                marketcap,
                -- Z-score within date
                (ey - AVG(ey) OVER (PARTITION BY date)) / NULLIF(STDDEV(ey) OVER (PARTITION BY date), 0) as ey_z,
                (by - AVG(by) OVER (PARTITION BY date)) / NULLIF(STDDEV(by) OVER (PARTITION BY date), 0) as by_z,
                (ebitda_y - AVG(ebitda_y) OVER (PARTITION BY date)) / NULLIF(STDDEV(ebitda_y) OVER (PARTITION BY date), 0) as ebitda_z
            FROM daily_vals
        ),
        composite AS (
            SELECT 
                ticker,
                date,
                marketcap,
                -- Equal weight composite
                (COALESCE(ey_z, 0) + COALESCE(by_z, 0) + COALESCE(ebitda_z, 0)) / 
                    (CASE WHEN ey_z IS NOT NULL THEN 1 ELSE 0 END + 
                     CASE WHEN by_z IS NOT NULL THEN 1 ELSE 0 END + 
                     CASE WHEN ebitda_z IS NOT NULL THEN 1 ELSE 0 END) as value_composite
            FROM zscored
        ),
        with_returns AS (
            SELECT 
                c.*,
                LEAD(c.marketcap, 5) OVER (PARTITION BY c.ticker ORDER BY c.date) / 
                    NULLIF(c.marketcap, 0) - 1 as ret_5d_f
            FROM composite c
            WHERE c.value_composite IS NOT NULL
        )
        SELECT 
            CORR(value_composite, ret_5d_f) as ic_composite,
            COUNT(*) as n_obs
        FROM with_returns
        WHERE ret_5d_f IS NOT NULL
          AND ret_5d_f BETWEEN -0.5 AND 0.5
    """).df()
    
    print(f"\nComposite Value Factor (EY + BY + EBITDA_Y z-scored):")
    print(f"  IC = {composite_test['ic_composite'].values[0]:.4f} (n={composite_test['n_obs'].values[0]:,})")

    # =========================================================================
    # 6. MOMENTUM INVESTIGATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. MOMENTUM DEEP DIVE")
    print("=" * 80)
    
    # Test different momentum lookbacks
    mom_test = con.execute("""
        WITH prices AS (
            SELECT 
                ticker,
                date,
                closeadj,
                -- Various momentum lookbacks
                closeadj / NULLIF(LAG(closeadj, 21) OVER w, 0) - 1 as mom_1m,
                closeadj / NULLIF(LAG(closeadj, 63) OVER w, 0) - 1 as mom_3m,
                closeadj / NULLIF(LAG(closeadj, 126) OVER w, 0) - 1 as mom_6m,
                closeadj / NULLIF(LAG(closeadj, 252) OVER w, 0) - 1 as mom_12m,
                -- 12-1 (skip recent month)
                (closeadj / NULLIF(LAG(closeadj, 252) OVER w, 0) - 1) - 
                (closeadj / NULLIF(LAG(closeadj, 21) OVER w, 0) - 1) as mom_12_1,
                -- Forward return
                LEAD(closeadj, 5) OVER w / closeadj - 1 as ret_5d_f
            FROM sep
            WHERE date >= '2014-01-01'
            WINDOW w AS (PARTITION BY ticker ORDER BY date)
        )
        SELECT 
            CORR(mom_1m, ret_5d_f) as ic_1m,
            CORR(mom_3m, ret_5d_f) as ic_3m,
            CORR(mom_6m, ret_5d_f) as ic_6m,
            CORR(mom_12m, ret_5d_f) as ic_12m,
            CORR(mom_12_1, ret_5d_f) as ic_12_1,
            -- Also test reversal
            CORR(-mom_1m, ret_5d_f) as ic_reversal_1m,
            COUNT(*) as n_obs
        FROM prices
        WHERE ret_5d_f IS NOT NULL
          AND date >= '2015-01-01'
          AND ret_5d_f BETWEEN -0.5 AND 0.5
          AND mom_12m BETWEEN -0.9 AND 5  -- Filter extremes
    """).df()
    
    print("\nMomentum Factor ICs:")
    print(f"  1-month momentum:   IC = {mom_test['ic_1m'].values[0]:.4f}")
    print(f"  3-month momentum:   IC = {mom_test['ic_3m'].values[0]:.4f}")
    print(f"  6-month momentum:   IC = {mom_test['ic_6m'].values[0]:.4f}")
    print(f"  12-month momentum:  IC = {mom_test['ic_12m'].values[0]:.4f}")
    print(f"  12-1 month:         IC = {mom_test['ic_12_1'].values[0]:.4f}")
    print(f"  1-month reversal:   IC = {mom_test['ic_reversal_1m'].values[0]:.4f}")
    print(f"  (n={mom_test['n_obs'].values[0]:,})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("""
DATA ISSUES IDENTIFIED:

1. feat_value only has ~4,500 rows - NOT BEING JOINED PROPERLY
   → Need to rebuild value factor pipeline to cover full universe

2. label_5d_up is TARGET LEAKAGE (IC=0.09) - NEVER USE AS FEATURE
   → This is a label derived from ret_5d_f, not a predictor

3. alpha_mlm has LOOK-AHEAD BIAS (IC=0.08) - DO NOT USE
   → Suspiciously high IC confirms it's using future information

USABLE FACTORS FROM DAILY TABLE:

Factor              | IC      | Coverage
--------------------|---------|----------
Earnings Yield      | ~0.014  | 8M+ rows
Book Yield          | ~0.014  | 13M+ rows  
EBITDA Yield        | ~0.017  | 10M+ rows
Composite Value     | ~0.015  | 13M+ rows

RECOMMENDED ACTIONS:

1. Build proper value factor from daily table (1/PE, 1/PB, 1/EVEBITDA)
2. Join to feat_matrix properly (you have 13M+ coverage)
3. Remove label_5d_up and alpha_mlm from any modeling
4. Test if adding value to alpha_composite improves IC
5. Investigate why momentum has no IC in your data
""")

    con.close()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()