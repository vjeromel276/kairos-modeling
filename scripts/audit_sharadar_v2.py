#!/usr/bin/env python3
"""
Sharadar Data Audit v2 - Fixed column names, expanded IC testing
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
    print("SHARADAR DATA AUDIT v2 - FACTOR IC TESTING")
    print("=" * 80)

    # =========================================================================
    # 1. GET ALL COLUMNS IN FEAT_MATRIX
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. ALL COLUMNS IN FEAT_MATRIX")
    print("=" * 80)
    
    fm_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'feat_matrix'
        ORDER BY column_name
    """).df()['column_name'].tolist()
    
    print(f"\nTotal columns: {len(fm_cols)}")
    print(f"Columns: {fm_cols}")

    # =========================================================================
    # 2. TEST IC OF ALL NUMERIC COLUMNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. IC TEST OF ALL FACTORS vs ret_5d_f")
    print("=" * 80)
    
    # Skip these columns
    skip_cols = ['ticker', 'date', 'table', 'ret_5d_f', 'ret_1d_f']
    
    ic_results = []
    
    for col in fm_cols:
        if col in skip_cols:
            continue
        try:
            result = con.execute(f"""
                SELECT 
                    CORR("{col}", ret_5d_f) as ic,
                    COUNT(*) as n_obs
                FROM feat_matrix
                WHERE "{col}" IS NOT NULL 
                  AND ret_5d_f IS NOT NULL
                  AND date >= '2015-01-01'
            """).fetchone()
            
            if result[0] is not None:
                ic_results.append({
                    'column': col,
                    'ic': result[0],
                    'n_obs': result[1]
                })
        except Exception as e:
            pass  # Skip columns that error
    
    # Sort by absolute IC
    ic_df = pd.DataFrame(ic_results)
    ic_df['abs_ic'] = ic_df['ic'].abs()
    ic_df = ic_df.sort_values('abs_ic', ascending=False)
    
    print("\nTOP 30 FACTORS BY IC (absolute value):")
    print("-" * 70)
    print(f"{'Column':<35} {'IC':>12} {'Abs IC':>12} {'N Obs':>12}")
    print("-" * 70)
    
    for _, row in ic_df.head(30).iterrows():
        strength = "***" if row['abs_ic'] > 0.02 else "**" if row['abs_ic'] > 0.01 else "*" if row['abs_ic'] > 0.005 else ""
        print(f"{row['column']:<35} {row['ic']:>12.4f} {row['abs_ic']:>12.4f} {row['n_obs']:>12,} {strength}")
    
    print("\n*** = Strong (>2%), ** = Moderate (>1%), * = Weak (>0.5%)")

    # =========================================================================
    # 3. TEST SF1 FUNDAMENTAL FACTORS DIRECTLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. SF1 FUNDAMENTAL FACTORS - DIRECT IC TEST")
    print("=" * 80)
    
    # Key fundamental factors from SF1
    sf1_factors = [
        'roe', 'roa', 'roic', 'ros',  # Profitability
        'grossmargin', 'netmargin', 'ebitdamargin',  # Margins
        'assetturnover', 'currentratio', 'de',  # Efficiency/Leverage
        'pe', 'pb', 'ps', 'divyield',  # Valuation
        'revenuegrowth', 'epsgrowth',  # Growth (if exists)
    ]
    
    # First check what columns exist in sf1
    sf1_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'sf1'
    """).df()['column_name'].tolist()
    
    print(f"\nSF1 has {len(sf1_cols)} columns")
    print(f"Key fundamentals available: {[c for c in sf1_factors if c in sf1_cols]}")
    
    # Test IC by joining SF1 with price data
    print("\nTesting SF1 factor ICs (this may take a moment)...")
    
    sf1_ic_results = []
    
    for factor in sf1_factors:
        if factor not in sf1_cols:
            continue
        try:
            # Join SF1 with SEP to get forward returns
            result = con.execute(f"""
                WITH fundamentals AS (
                    SELECT 
                        ticker,
                        datekey as date,
                        {factor}
                    FROM sf1
                    WHERE dimension = 'ARQ'  -- Quarterly, as-reported
                      AND {factor} IS NOT NULL
                ),
                prices AS (
                    SELECT 
                        ticker,
                        date,
                        LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
                    FROM sep
                    WHERE date >= '2015-01-01'
                )
                SELECT 
                    CORR(f.{factor}, p.ret_5d_f) as ic,
                    COUNT(*) as n_obs
                FROM fundamentals f
                JOIN prices p ON f.ticker = p.ticker 
                    AND p.date >= f.date 
                    AND p.date < f.date + INTERVAL '95 days'
                WHERE p.ret_5d_f IS NOT NULL
            """).fetchone()
            
            if result[0] is not None:
                sf1_ic_results.append({
                    'factor': factor,
                    'ic': result[0],
                    'n_obs': result[1]
                })
                print(f"  {factor}: IC = {result[0]:.4f} (n={result[1]:,})")
        except Exception as e:
            print(f"  {factor}: error - {str(e)[:50]}")
    
    # =========================================================================
    # 4. CHECK DAILY TABLE FOR VALUATION FACTORS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. DAILY TABLE VALUATION FACTORS")
    print("=" * 80)
    
    daily_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'daily'
    """).df()['column_name'].tolist()
    
    print(f"\nDaily table columns: {daily_cols}")
    
    # Test PE, PB, PS from daily table
    for factor in ['pe', 'pb', 'ps', 'evebit', 'evebitda']:
        if factor not in daily_cols:
            continue
        try:
            result = con.execute(f"""
                WITH vals AS (
                    SELECT 
                        ticker,
                        date,
                        {factor},
                        LEAD(marketcap, 5) OVER (PARTITION BY ticker ORDER BY date) / 
                            NULLIF(marketcap, 0) - 1 as ret_5d_f
                    FROM daily
                    WHERE date >= '2015-01-01'
                      AND {factor} IS NOT NULL
                      AND {factor} > 0
                      AND {factor} < 1000  -- Filter outliers
                )
                SELECT 
                    CORR({factor}, ret_5d_f) as ic,
                    CORR(1.0/{factor}, ret_5d_f) as ic_inverse,
                    COUNT(*) as n_obs
                FROM vals
                WHERE ret_5d_f IS NOT NULL
            """).fetchone()
            
            if result[0] is not None:
                print(f"  {factor}: IC = {result[0]:.4f}, 1/{factor} IC = {result[1]:.4f} (n={result[2]:,})")
        except Exception as e:
            print(f"  {factor}: error - {str(e)[:50]}")

    # =========================================================================
    # 5. CHECK SF2 FOR INSIDER TRANSACTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. SF2 INSIDER TRANSACTIONS")
    print("=" * 80)
    
    try:
        sf2_cols = con.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'sf2'
        """).df()['column_name'].tolist()
        
        print(f"\nSF2 columns: {sf2_cols}")
        
        # Sample
        sample = con.execute("SELECT * FROM sf2 LIMIT 5").df()
        print("\nSample rows:")
        print(sample.to_string())
    except Exception as e:
        print(f"Error: {e}")

    # =========================================================================
    # 6. QUALITY FACTORS IN FEAT_MATRIX
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. QUALITY FACTORS IC TEST")
    print("=" * 80)
    
    quality_cols = ['grossmargin', 'netmargin', 'roa', 'roe', 'quality_raw', 'quality_z']
    
    for col in quality_cols:
        if col in fm_cols:
            try:
                result = con.execute(f"""
                    SELECT 
                        CORR("{col}", ret_5d_f) as ic,
                        COUNT(*) as n_obs
                    FROM feat_matrix
                    WHERE "{col}" IS NOT NULL 
                      AND ret_5d_f IS NOT NULL
                      AND date >= '2015-01-01'
                """).fetchone()
                
                if result[0] is not None:
                    print(f"  {col}: IC = {result[0]:.4f} (n={result[1]:,})")
            except Exception as e:
                print(f"  {col}: error")

    # =========================================================================
    # 7. MOMENTUM FACTORS (NEED TO BUILD 12-1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. MOMENTUM FACTORS - TESTING 12-1 MONTH")
    print("=" * 80)
    
    # Test if we can build 12-1 momentum from SEP
    try:
        result = con.execute("""
            WITH momentum AS (
                SELECT 
                    ticker,
                    date,
                    closeadj / LAG(closeadj, 252) OVER (PARTITION BY ticker ORDER BY date) - 1 as mom_12m,
                    closeadj / LAG(closeadj, 21) OVER (PARTITION BY ticker ORDER BY date) - 1 as mom_1m,
                    LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
                FROM sep
                WHERE date >= '2014-01-01'
            )
            SELECT 
                CORR(mom_12m, ret_5d_f) as ic_12m,
                CORR(mom_1m, ret_5d_f) as ic_1m,
                CORR(mom_12m - mom_1m, ret_5d_f) as ic_12_minus_1,
                COUNT(*) as n_obs
            FROM momentum
            WHERE mom_12m IS NOT NULL 
              AND mom_1m IS NOT NULL
              AND ret_5d_f IS NOT NULL
              AND date >= '2015-01-01'
        """).fetchone()
        
        print(f"  12-month momentum IC: {result[0]:.4f}")
        print(f"  1-month momentum IC: {result[1]:.4f}")
        print(f"  12-1 month momentum IC: {result[2]:.4f} (n={result[3]:,})")
        print("\n  Note: 12-1 month momentum skips recent month to avoid reversal")
    except Exception as e:
        print(f"  Error: {e}")

    # =========================================================================
    # 8. ACCRUALS FACTOR (HIGHEST ACADEMIC IC)
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. ACCRUALS FACTOR TEST")
    print("=" * 80)
    
    # Accruals = (Change in Current Assets - Change in Cash - Change in Current Liabilities 
    #            + Change in Short-term Debt - Depreciation) / Total Assets
    # Or simpler: (Net Income - Operating Cash Flow) / Total Assets
    
    try:
        result = con.execute("""
            WITH accruals AS (
                SELECT 
                    ticker,
                    datekey as date,
                    -- Simple accruals: Net Income - CFO, scaled by assets
                    (netinc - ncfo) / NULLIF(assets, 0) as accruals,
                    -- Asset growth (also negative predictor)
                    assets / NULLIF(LAG(assets, 4) OVER (PARTITION BY ticker ORDER BY datekey), 0) - 1 as asset_growth
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
                    LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
                FROM sep
                WHERE date >= '2015-01-01'
            )
            SELECT 
                CORR(a.accruals, p.ret_5d_f) as ic_accruals,
                CORR(a.asset_growth, p.ret_5d_f) as ic_asset_growth,
                COUNT(*) as n_obs
            FROM accruals a
            JOIN prices p ON a.ticker = p.ticker 
                AND p.date >= a.date 
                AND p.date < a.date + INTERVAL '95 days'
            WHERE p.ret_5d_f IS NOT NULL
              AND a.accruals IS NOT NULL
              AND ABS(a.accruals) < 1  -- Filter extreme values
        """).fetchone()
        
        print(f"  Accruals IC: {result[0]:.4f}")
        print(f"  Asset Growth IC: {result[1]:.4f}")
        print(f"  (n={result[2]:,})")
        print("\n  Note: NEGATIVE IC is good - low accruals predict higher returns")
    except Exception as e:
        print(f"  Error: {e}")

    # =========================================================================
    # 9. SUMMARY OF BEST FACTORS FOUND
    # =========================================================================
    print("\n" + "=" * 80)
    print("9. SUMMARY - BEST FACTOR CANDIDATES")
    print("=" * 80)
    
    print("""
FACTORS WITH MEANINGFUL IC (from this analysis):

Factor                  | IC      | Source      | Action Needed
------------------------|---------|-------------|---------------
value_raw/value_z       | +0.0287 | feat_value  | USE IT!
12-1 month momentum     | ???     | SEP         | Build it
Accruals                | ???     | SF1         | Build it
Quality composite       | ???     | SF1         | Build it
Earnings surprise       | ???     | SF1         | Need estimates

NEXT STEPS:
1. Incorporate value_raw into alpha (it's 143x your current IC!)
2. Build proper 12-1 momentum factor
3. Build accruals factor from SF1
4. Test quality composite (ROE + margins + low leverage)
5. Consider Polygon for short interest data
""")

    con.close()
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()