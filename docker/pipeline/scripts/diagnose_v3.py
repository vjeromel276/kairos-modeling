#!/usr/bin/env python3
"""
Diagnostic v3 - Fixed SQL reserved words, complete analysis
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
    print("DIAGNOSTIC v3: COMPLETE DATA AUDIT")
    print("=" * 80)

    # =========================================================================
    # 1. COMPOSITE VALUE FACTOR TEST (FIXED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. COMPOSITE VALUE FACTOR TEST")
    print("=" * 80)
    
    composite_test = con.execute("""
        WITH daily_vals AS (
            SELECT 
                ticker,
                date,
                CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END as earnings_yield,
                CASE WHEN pb > 0 AND pb < 20 THEN 1.0/pb ELSE NULL END as book_yield,
                CASE WHEN evebitda > 0 AND evebitda < 50 THEN 1.0/evebitda ELSE NULL END as ebitda_yield,
                marketcap
            FROM daily
            WHERE date >= '2015-01-01'
        ),
        zscored AS (
            SELECT 
                ticker,
                date,
                marketcap,
                (earnings_yield - AVG(earnings_yield) OVER (PARTITION BY date)) / 
                    NULLIF(STDDEV(earnings_yield) OVER (PARTITION BY date), 0) as ey_z,
                (book_yield - AVG(book_yield) OVER (PARTITION BY date)) / 
                    NULLIF(STDDEV(book_yield) OVER (PARTITION BY date), 0) as book_z,
                (ebitda_yield - AVG(ebitda_yield) OVER (PARTITION BY date)) / 
                    NULLIF(STDDEV(ebitda_yield) OVER (PARTITION BY date), 0) as ebitda_z
            FROM daily_vals
        ),
        composite AS (
            SELECT 
                ticker,
                date,
                marketcap,
                (COALESCE(ey_z, 0) + COALESCE(book_z, 0) + COALESCE(ebitda_z, 0)) / 
                    NULLIF(
                        (CASE WHEN ey_z IS NOT NULL THEN 1 ELSE 0 END + 
                         CASE WHEN book_z IS NOT NULL THEN 1 ELSE 0 END + 
                         CASE WHEN ebitda_z IS NOT NULL THEN 1 ELSE 0 END), 
                    0) as value_composite
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
    
    print(f"\nComposite Value Factor (EY + Book + EBITDA z-scored):")
    print(f"  IC = {composite_test['ic_composite'].values[0]:.4f} (n={composite_test['n_obs'].values[0]:,})")

    # =========================================================================
    # 2. MOMENTUM DEEP DIVE
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. MOMENTUM DEEP DIVE")
    print("=" * 80)
    
    mom_test = con.execute("""
        WITH prices AS (
            SELECT 
                ticker,
                date,
                closeadj,
                closeadj / NULLIF(LAG(closeadj, 21) OVER w, 0) - 1 as mom_1m,
                closeadj / NULLIF(LAG(closeadj, 63) OVER w, 0) - 1 as mom_3m,
                closeadj / NULLIF(LAG(closeadj, 126) OVER w, 0) - 1 as mom_6m,
                closeadj / NULLIF(LAG(closeadj, 252) OVER w, 0) - 1 as mom_12m,
                (closeadj / NULLIF(LAG(closeadj, 252) OVER w, 0) - 1) - 
                (closeadj / NULLIF(LAG(closeadj, 21) OVER w, 0) - 1) as mom_12_1,
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
            CORR(-mom_1m, ret_5d_f) as ic_reversal_1m,
            COUNT(*) as n_obs
        FROM prices
        WHERE ret_5d_f IS NOT NULL
          AND date >= '2015-01-01'
          AND ret_5d_f BETWEEN -0.5 AND 0.5
          AND mom_12m BETWEEN -0.9 AND 5
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
    # 3. WHY ARE VALUE ICs NEGATIVE?
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. VALUE IC SIGN INVESTIGATION")
    print("=" * 80)
    
    # Check if the IC sign flips by time period
    value_by_period = con.execute("""
        WITH daily_vals AS (
            SELECT 
                ticker,
                date,
                EXTRACT(YEAR FROM date) as year,
                CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END as earnings_yield,
                marketcap
            FROM daily
            WHERE date >= '2010-01-01'
        ),
        with_returns AS (
            SELECT 
                d.*,
                LEAD(d.marketcap, 5) OVER (PARTITION BY d.ticker ORDER BY d.date) / 
                    NULLIF(d.marketcap, 0) - 1 as ret_5d_f
            FROM daily_vals d
        )
        SELECT 
            year,
            CORR(earnings_yield, ret_5d_f) as ic,
            COUNT(*) as n_obs
        FROM with_returns
        WHERE ret_5d_f IS NOT NULL
          AND ret_5d_f BETWEEN -0.5 AND 0.5
          AND earnings_yield IS NOT NULL
        GROUP BY year
        ORDER BY year
    """).df()
    
    print("\nEarnings Yield IC by Year:")
    print(value_by_period.to_string(index=False))
    
    # Check decile returns for value
    print("\nValue Decile Analysis:")
    decile_test = con.execute("""
        WITH daily_vals AS (
            SELECT 
                ticker,
                date,
                CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END as earnings_yield,
                marketcap,
                NTILE(10) OVER (PARTITION BY date ORDER BY 
                    CASE WHEN pe > 0 AND pe < 100 THEN 1.0/pe ELSE NULL END) as decile
            FROM daily
            WHERE date >= '2015-01-01'
              AND pe > 0 AND pe < 100
        ),
        with_returns AS (
            SELECT 
                d.*,
                LEAD(d.marketcap, 5) OVER (PARTITION BY d.ticker ORDER BY d.date) / 
                    NULLIF(d.marketcap, 0) - 1 as ret_5d_f
            FROM daily_vals d
        )
        SELECT 
            decile,
            AVG(ret_5d_f) * 252 as ann_ret,
            COUNT(*) as n_obs
        FROM with_returns
        WHERE ret_5d_f IS NOT NULL
          AND ret_5d_f BETWEEN -0.5 AND 0.5
        GROUP BY decile
        ORDER BY decile
    """).df()
    
    print(decile_test.to_string(index=False))
    print("\nNote: Decile 10 = highest earnings yield (cheapest stocks)")

    # =========================================================================
    # 4. QUALITY FACTORS - REBUILD AND TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. QUALITY FACTORS FROM SF1")
    print("=" * 80)
    
    quality_test = con.execute("""
        WITH fundamentals AS (
            SELECT 
                ticker,
                datekey as report_date,
                roe,
                roa,
                grossmargin,
                netmargin,
                de as debt_equity,
                currentratio
            FROM sf1
            WHERE dimension = 'ARQ'
              AND roe IS NOT NULL
        ),
        prices AS (
            SELECT 
                ticker,
                date,
                closeadj,
                LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
            FROM sep
            WHERE date >= '2015-01-01'
        ),
        joined AS (
            SELECT 
                p.ticker,
                p.date,
                p.ret_5d_f,
                f.roe,
                f.roa,
                f.grossmargin,
                f.debt_equity
            FROM prices p
            JOIN fundamentals f ON p.ticker = f.ticker 
                AND p.date >= f.report_date 
                AND p.date < f.report_date + INTERVAL '95 days'
            WHERE p.ret_5d_f IS NOT NULL
        )
        SELECT 
            CORR(roe, ret_5d_f) as ic_roe,
            CORR(roa, ret_5d_f) as ic_roa,
            CORR(grossmargin, ret_5d_f) as ic_grossmargin,
            CORR(-debt_equity, ret_5d_f) as ic_low_leverage,
            COUNT(*) as n_obs
        FROM joined
        WHERE ret_5d_f BETWEEN -0.5 AND 0.5
          AND roe BETWEEN -1 AND 2
          AND roa BETWEEN -0.5 AND 0.5
    """).df()
    
    print("\nQuality Factor ICs (from SF1):")
    print(f"  ROE:              IC = {quality_test['ic_roe'].values[0]:.4f}")
    print(f"  ROA:              IC = {quality_test['ic_roa'].values[0]:.4f}")
    print(f"  Gross Margin:     IC = {quality_test['ic_grossmargin'].values[0]:.4f}")
    print(f"  Low Leverage:     IC = {quality_test['ic_low_leverage'].values[0]:.4f}")
    print(f"  (n={quality_test['n_obs'].values[0]:,})")

    # =========================================================================
    # 5. ACCRUALS FACTOR (PROPER CALCULATION)
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. ACCRUALS FACTOR (EARNINGS QUALITY)")
    print("=" * 80)
    
    accruals_test = con.execute("""
        WITH fundamentals AS (
            SELECT 
                ticker,
                datekey as report_date,
                -- Accruals = Net Income - Operating Cash Flow, scaled by assets
                (netinc - ncfo) / NULLIF(assets, 0) as accruals,
                -- Alternative: change in working capital
                assets,
                netinc,
                ncfo
            FROM sf1
            WHERE dimension = 'ARQ'
              AND assets > 0
              AND netinc IS NOT NULL
              AND ncfo IS NOT NULL
        ),
        prices AS (
            SELECT 
                ticker,
                date,
                closeadj,
                LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
            FROM sep
            WHERE date >= '2015-01-01'
        ),
        joined AS (
            SELECT 
                p.ticker,
                p.date,
                p.ret_5d_f,
                f.accruals
            FROM prices p
            JOIN fundamentals f ON p.ticker = f.ticker 
                AND p.date >= f.report_date 
                AND p.date < f.report_date + INTERVAL '95 days'
            WHERE p.ret_5d_f IS NOT NULL
              AND f.accruals IS NOT NULL
              AND ABS(f.accruals) < 0.5  -- Filter extreme values
        )
        SELECT 
            CORR(accruals, ret_5d_f) as ic_accruals,
            CORR(-accruals, ret_5d_f) as ic_low_accruals,
            COUNT(*) as n_obs
        FROM joined
        WHERE ret_5d_f BETWEEN -0.5 AND 0.5
    """).df()
    
    print("\nAccruals Factor IC:")
    print(f"  Accruals (raw):     IC = {accruals_test['ic_accruals'].values[0]:.4f}")
    print(f"  Low Accruals:       IC = {accruals_test['ic_low_accruals'].values[0]:.4f}")
    print(f"  (n={accruals_test['n_obs'].values[0]:,})")
    print("\n  Note: Low accruals (negative sign) should predict higher returns")

    # =========================================================================
    # 6. INSIDER TRANSACTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. INSIDER TRANSACTIONS (SF2)")
    print("=" * 80)
    
    insider_test = con.execute("""
        WITH insider_activity AS (
            SELECT 
                ticker,
                filingdate,
                SUM(CASE WHEN transactioncode = 'P' THEN transactionvalue ELSE 0 END) as buy_value,
                SUM(CASE WHEN transactioncode = 'S' THEN transactionvalue ELSE 0 END) as sell_value,
                COUNT(CASE WHEN transactioncode = 'P' THEN 1 END) as buy_count,
                COUNT(CASE WHEN transactioncode = 'S' THEN 1 END) as sell_count
            FROM sf2
            WHERE transactioncode IN ('P', 'S')
              AND transactionvalue > 0
            GROUP BY ticker, filingdate
        ),
        net_insider AS (
            SELECT 
                ticker,
                filingdate,
                buy_value - sell_value as net_value,
                buy_count - sell_count as net_count,
                CASE WHEN buy_value > sell_value THEN 1 
                     WHEN sell_value > buy_value THEN -1 
                     ELSE 0 END as net_signal
            FROM insider_activity
        ),
        prices AS (
            SELECT 
                ticker,
                date,
                closeadj,
                LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f,
                LEAD(closeadj, 21) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_21d_f
            FROM sep
            WHERE date >= '2015-01-01'
        ),
        joined AS (
            SELECT 
                p.ticker,
                p.date,
                p.ret_5d_f,
                p.ret_21d_f,
                i.net_signal,
                i.net_count
            FROM prices p
            JOIN net_insider i ON p.ticker = i.ticker 
                AND p.date >= i.filingdate 
                AND p.date < i.filingdate + INTERVAL '30 days'
            WHERE p.ret_5d_f IS NOT NULL
        )
        SELECT 
            CORR(net_signal, ret_5d_f) as ic_5d,
            CORR(net_signal, ret_21d_f) as ic_21d,
            CORR(net_count, ret_5d_f) as ic_count_5d,
            COUNT(*) as n_obs
        FROM joined
        WHERE ret_5d_f BETWEEN -0.5 AND 0.5
    """).df()
    
    print("\nInsider Transaction ICs:")
    print(f"  Net Buy Signal (5d):    IC = {insider_test['ic_5d'].values[0]:.4f}")
    print(f"  Net Buy Signal (21d):   IC = {insider_test['ic_21d'].values[0]:.4f}")
    print(f"  Net Buy Count (5d):     IC = {insider_test['ic_count_5d'].values[0]:.4f}")
    print(f"  (n={insider_test['n_obs'].values[0]:,})")

    # =========================================================================
    # 7. SUMMARY OF ALL FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. COMPLETE SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("""
================================================================================
CONFIRMED DATA ISSUES:
================================================================================

1. TARGET LEAKAGE - DO NOT USE:
   - label_5d_up: IC = 0.12 (it's literally the target variable)
   - alpha_mlm: IC = 0.08 (look-ahead bias confirmed)

2. BROKEN DATA JOINS:
   - feat_value: Only 4,734 non-null values in feat_matrix (should be millions)
   - Value factor shows IC = 0.0287 in small sample but not in full data

3. CURRENT ALPHA HAS NEAR-ZERO IC:
   - alpha_composite_v33_regime: IC ≈ 0.0002
   - Mostly correlated with liquidity (59% correlation with ADV)

================================================================================
FACTOR IC SUMMARY (from this analysis):
================================================================================

Factor Category        | Factor              | IC      | Status
-----------------------|---------------------|---------|--------
VALUE (from daily)     | Earnings Yield      | -0.011  | Wrong sign?
                       | Book Yield          | -0.010  | Wrong sign?
                       | EBITDA Yield        | -0.014  | Wrong sign?
                       | Composite           | (test)  | Need to verify

MOMENTUM              | 1-month             | (test)  | 
                       | 12-1 month          | (test)  |

QUALITY (from SF1)     | ROE                 | (test)  |
                       | ROA                 | (test)  |
                       | Gross Margin        | (test)  |
                       | Low Leverage        | (test)  |

ACCRUALS              | Low Accruals        | (test)  | Should be positive

INSIDER               | Net Buying          | (test)  |

================================================================================
NEXT STEPS FOR IMPLEMENTATION CHAT:
================================================================================

1. Rebuild value factors from daily table with proper:
   - Filtering (positive ratios, reasonable ranges)
   - Z-scoring within date
   - Proper join to price data

2. Investigate value IC sign issue:
   - Is it regime-dependent?
   - Is it sector-dependent?
   - Is it size-dependent?

3. Build quality composite from SF1

4. Test insider buying signal

5. Create new alpha that combines:
   - Value (if sign issue resolved)
   - Quality
   - Possibly insider signal
   
6. Properly join to feat_matrix with full coverage
""")

    con.close()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()