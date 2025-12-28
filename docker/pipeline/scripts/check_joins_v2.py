#!/usr/bin/env python3
"""
Check all fundamental data joins to identify broken pipelines - FIXED
"""

import argparse
import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("FUNDAMENTAL DATA JOIN DIAGNOSTIC")
    print("=" * 80)

    # =========================================================================
    # 1. CHECK SF1 TABLE STRUCTURE
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. SF1 TABLE STRUCTURE")
    print("=" * 80)
    
    sf1_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT datekey) as unique_datekeys,
            MIN(datekey) as min_date,
            MAX(datekey) as max_date,
            COUNT(DISTINCT dimension) as dimensions
        FROM sf1
    """).df()
    print("\nSF1 summary:")
    print(sf1_info.to_string(index=False))
    
    # Check dimensions
    dims = con.execute("""
        SELECT dimension, COUNT(*) as cnt 
        FROM sf1 
        GROUP BY dimension
    """).df()
    print("\nSF1 dimensions:")
    print(dims.to_string(index=False))
    
    # Check date columns
    print("\nSF1 date column samples:")
    date_sample = con.execute("""
        SELECT ticker, dimension, calendardate, datekey, reportperiod 
        FROM sf1 
        WHERE dimension = 'ARQ'
        LIMIT 5
    """).df()
    print(date_sample.to_string(index=False))

    # =========================================================================
    # 2. CHECK SEP TABLE STRUCTURE  
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. SEP TABLE STRUCTURE")
    print("=" * 80)
    
    sep_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM sep
    """).df()
    print("\nSEP summary:")
    print(sep_info.to_string(index=False))
    
    # Check date type separately
    date_type = con.execute("SELECT typeof(date) as date_type FROM sep LIMIT 1").df()
    print(f"SEP date type: {date_type['date_type'].values[0]}")
    
    # Sample
    print("\nSEP sample:")
    sep_sample = con.execute("SELECT * FROM sep LIMIT 3").df()
    print(sep_sample.to_string(index=False))

    # =========================================================================
    # 3. TEST SF1 -> SEP JOIN
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. SF1 -> SEP JOIN TEST")
    print("=" * 80)
    
    # Check if tickers overlap
    ticker_overlap = con.execute("""
        SELECT 
            (SELECT COUNT(DISTINCT ticker) FROM sf1) as sf1_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM sep) as sep_tickers,
            (SELECT COUNT(DISTINCT sf1.ticker) 
             FROM sf1 
             JOIN sep ON sf1.ticker = sep.ticker) as overlapping_tickers
    """).df()
    print("\nTicker overlap:")
    print(ticker_overlap.to_string(index=False))
    
    # Check datekey type
    print("\nChecking datekey type in SF1:")
    datekey_type = con.execute("SELECT typeof(datekey) as type FROM sf1 LIMIT 1").df()
    print(f"SF1 datekey type: {datekey_type['type'].values[0]}")
    
    # Check if datekey is actually a date or string
    print("\nDatekey format check:")
    datekey_check = con.execute("""
        SELECT 
            datekey,
            TRY_CAST(datekey AS DATE) as as_date
        FROM sf1 
        WHERE datekey IS NOT NULL
        LIMIT 5
    """).df()
    print(datekey_check.to_string(index=False))

    # =========================================================================
    # 4. TEST DAILY TABLE JOINS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. DAILY TABLE JOIN TEST")
    print("=" * 80)
    
    daily_info = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM daily
    """).df()
    print("\nDaily summary:")
    print(daily_info.to_string(index=False))
    
    # Check date type
    daily_date_type = con.execute("SELECT typeof(date) as date_type FROM daily LIMIT 1").df()
    print(f"Daily date type: {daily_date_type['date_type'].values[0]}")

    # =========================================================================
    # 5. CHECK FEAT_MATRIX COLUMN NULL RATES
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. FEAT_MATRIX COLUMN NULL RATES")
    print("=" * 80)
    
    # Get all columns
    fm_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'feat_matrix'
    """).df()['column_name'].tolist()
    
    # Check null rates for key columns
    key_cols = ['value_raw', 'value_z', 'quality_raw', 'quality_z', 'roe', 'roa', 
                'grossmargin', 'netmargin', 'inst_value', 'inst_shares',
                'alpha_composite_v33_regime', 'ret_5d_f', 'adv_20']
    
    print(f"\n{'Column':<30} {'Non-Null':>12} {'Total':>12} {'Coverage':>10}")
    print("-" * 70)
    
    total_rows = con.execute("SELECT COUNT(*) FROM feat_matrix").fetchone()[0]
    
    for col in key_cols:
        if col in fm_cols:
            try:
                non_null = con.execute(f'SELECT COUNT("{col}") FROM feat_matrix').fetchone()[0]
                coverage = 100 * non_null / total_rows
                print(f"{col:<30} {non_null:>12,} {total_rows:>12,} {coverage:>9.1f}%")
            except:
                print(f"{col:<30} ERROR")
        else:
            print(f"{col:<30} NOT FOUND")

    # =========================================================================
    # 6. DEBUGGING THE QUALITY JOIN - STEP BY STEP
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. DEBUGGING THE QUALITY JOIN")
    print("=" * 80)
    
    # Step 1: Check SF1 data exists with filters
    print("\nStep 1: Check SF1 data exists with filters...")
    sf1_check = con.execute("""
        SELECT COUNT(*) as cnt
        FROM sf1
        WHERE dimension = 'ARQ'
          AND roe IS NOT NULL
    """).fetchone()[0]
    print(f"  SF1 rows with ARQ + ROE: {sf1_check:,}")
    
    # Step 2: Check ROE distribution
    print("\nStep 2: Check ROE distribution in SF1...")
    roe_dist = con.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(roe) as has_roe,
            AVG(roe) as avg_roe,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roe) as median_roe,
            MIN(roe) as min_roe,
            MAX(roe) as max_roe
        FROM sf1
        WHERE dimension = 'ARQ'
    """).df()
    print(roe_dist.to_string(index=False))
    
    # Step 3: Simple ticker match test
    print("\nStep 3: Simple ticker match test...")
    ticker_test = con.execute("""
        SELECT COUNT(DISTINCT f.ticker) as matching_tickers
        FROM sf1 f
        WHERE f.dimension = 'ARQ'
          AND EXISTS (SELECT 1 FROM sep s WHERE s.ticker = f.ticker)
    """).fetchone()[0]
    print(f"  SF1 ARQ tickers that exist in SEP: {ticker_test:,}")
    
    # Step 4: Test a simpler join
    print("\nStep 4: Testing simpler join (same date)...")
    simple_join = con.execute("""
        SELECT COUNT(*) as cnt
        FROM sf1 f
        JOIN sep p ON f.ticker = p.ticker 
            AND CAST(p.date AS DATE) = CAST(f.datekey AS DATE)
        WHERE f.dimension = 'ARQ'
          AND f.roe IS NOT NULL
        LIMIT 100000
    """).fetchone()[0]
    print(f"  Exact date match join count: {simple_join:,}")
    
    # Step 5: Test with date range
    print("\nStep 5: Testing with date range...")
    range_join = con.execute("""
        SELECT COUNT(*) as cnt
        FROM sf1 f
        JOIN sep p ON f.ticker = p.ticker 
        WHERE f.dimension = 'ARQ'
          AND f.roe IS NOT NULL
          AND CAST(p.date AS DATE) >= CAST(f.datekey AS DATE)
          AND CAST(p.date AS DATE) <= CAST(f.datekey AS DATE) + INTERVAL '90 days'
        LIMIT 100000
    """).fetchone()[0]
    print(f"  Date range join count (limited): {range_join:,}")
    
    # Step 6: Check what ret_5d_f looks like
    print("\nStep 6: Check forward return calculation...")
    ret_check = con.execute("""
        SELECT 
            ticker,
            date,
            closeadj,
            LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) as closeadj_5d,
            LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) / closeadj - 1 as ret_5d_f
        FROM sep
        WHERE ticker = 'AAPL'
          AND date >= '2020-01-01'
        LIMIT 10
    """).df()
    print(ret_check.to_string(index=False))

    # =========================================================================
    # 7. CHECK FEAT_ TABLE COVERAGE
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. FEAT_ TABLE ROW COUNTS")
    print("=" * 80)
    
    feat_tables = [
        'feat_value', 'feat_institutional_academic', 'feat_composite_academic',
        'feat_price_action', 'feat_targets', 'feat_matrix'
    ]
    
    print(f"\n{'Table':<35} {'Rows':>15}")
    print("-" * 55)
    
    for table in feat_tables:
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:<35} {cnt:>15,}")
        except Exception as e:
            print(f"{table:<35} ERROR: {str(e)[:30]}")

    # =========================================================================
    # 8. THE ACTUAL PROBLEM - VALUE TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. FEAT_VALUE DEEP DIVE")
    print("=" * 80)
    
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
    
    # Check if it's one row per ticker (snapshot) vs time series
    print("\nRows per ticker:")
    rows_per_ticker = con.execute("""
        SELECT ticker, COUNT(*) as cnt
        FROM feat_value
        GROUP BY ticker
        ORDER BY cnt DESC
        LIMIT 10
    """).df()
    print(rows_per_ticker.to_string(index=False))
    
    print("\nDates in feat_value:")
    dates_check = con.execute("""
        SELECT date, COUNT(*) as cnt
        FROM feat_value
        GROUP BY date
        ORDER BY date DESC
        LIMIT 10
    """).df()
    print(dates_check.to_string(index=False))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("""
KEY FINDINGS:

1. SF1 has 663,888 ARQ (quarterly as-reported) rows
2. SEP has millions of price rows
3. Tickers overlap between SF1 and SEP
4. Date types need to be checked for compatibility

LIKELY ISSUES:

1. feat_value appears to be a SNAPSHOT (one date per ticker)
   rather than a TIME SERIES - this is why coverage is low

2. The quality join may have issues with:
   - Date type casting
   - Filter ranges being too restrictive
   - Forward return calculation returning NULL

CHECK THE OUTPUT ABOVE TO IDENTIFY THE SPECIFIC ISSUE.
""")

    con.close()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()