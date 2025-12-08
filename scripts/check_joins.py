#!/usr/bin/env python3
"""
Check all fundamental data joins to identify broken pipelines
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
            MAX(date) as max_date,
            typeof(date) as date_type
        FROM sep
    """).df()
    print("\nSEP summary:")
    print(sep_info.to_string(index=False))
    
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
    
    # Test the actual join that failed
    print("\nTesting SF1 -> SEP join with date range...")
    
    join_test = con.execute("""
        WITH fundamentals AS (
            SELECT 
                ticker,
                datekey as report_date,
                roe,
                roa
            FROM sf1
            WHERE dimension = 'ARQ'
              AND roe IS NOT NULL
              AND datekey >= '2015-01-01'
            LIMIT 1000
        ),
        prices AS (
            SELECT 
                ticker,
                date,
                closeadj
            FROM sep
            WHERE date >= '2015-01-01'
        )
        SELECT COUNT(*) as join_count
        FROM fundamentals f
        JOIN prices p ON f.ticker = p.ticker 
            AND p.date >= f.report_date 
            AND p.date < f.report_date + INTERVAL '95 days'
    """).df()
    print(f"Join count (limited test): {join_test['join_count'].values[0]}")
    
    # Check datekey type
    print("\nChecking datekey type in SF1:")
    datekey_type = con.execute("""
        SELECT typeof(datekey) as type, datekey 
        FROM sf1 
        LIMIT 1
    """).df()
    print(datekey_type.to_string(index=False))
    
    # Check if datekey is actually a date or string
    print("\nDatekey format check:")
    datekey_check = con.execute("""
        SELECT 
            datekey,
            TRY_CAST(datekey AS DATE) as as_date,
            typeof(datekey) as original_type
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
    
    daily_join = con.execute("""
        SELECT 
            (SELECT COUNT(DISTINCT ticker) FROM daily) as daily_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM sep) as sep_tickers,
            (SELECT COUNT(*) 
             FROM daily d
             JOIN sep s ON d.ticker = s.ticker AND CAST(d.date AS DATE) = CAST(s.date AS DATE)
             WHERE d.date >= '2020-01-01'
             LIMIT 1000000) as join_sample
    """).df()
    print("\nDaily -> SEP join:")
    print(daily_join.to_string(index=False))

    # =========================================================================
    # 5. CHECK ALL FEAT_ TABLES FOR NULL RATES
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. FEAT_ TABLES NULL RATE CHECK")
    print("=" * 80)
    
    feat_tables = con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE 'feat_%'
        ORDER BY table_name
    """).df()['table_name'].tolist()
    
    print(f"\n{'Table':<35} {'Rows':>12} {'Key Cols Non-Null':>20}")
    print("-" * 70)
    
    for table in feat_tables[:20]:  # Limit to first 20
        try:
            # Get row count and check a few key columns
            info = con.execute(f"""
                SELECT 
                    COUNT(*) as rows,
                    COUNT(ticker) as ticker_nonnull,
                    COUNT(date) as date_nonnull
                FROM {table}
            """).fetchone()
            print(f"{table:<35} {info[0]:>12,} {info[1]:>10,}/{info[2]:>8,}")
        except Exception as e:
            print(f"{table:<35} ERROR: {str(e)[:30]}")

    # =========================================================================
    # 6. CHECK FEAT_MATRIX JOIN QUALITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. FEAT_MATRIX COLUMN NULL RATES")
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
    # 7. IDENTIFY THE ACTUAL JOIN ISSUE
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. DEBUGGING THE QUALITY JOIN")
    print("=" * 80)
    
    # Step by step debugging
    print("\nStep 1: Check SF1 data exists with filters...")
    sf1_check = con.execute("""
        SELECT COUNT(*) as cnt
        FROM sf1
        WHERE dimension = 'ARQ'
          AND roe IS NOT NULL
          AND datekey >= '2015-01-01'
    """).fetchone()[0]
    print(f"  SF1 rows with ARQ + ROE + date >= 2015: {sf1_check:,}")
    
    print("\nStep 2: Check date format compatibility...")
    date_compat = con.execute("""
        SELECT 
            sf1.ticker,
            sf1.datekey as sf1_date,
            typeof(sf1.datekey) as sf1_type,
            sep.date as sep_date,
            typeof(sep.date) as sep_type
        FROM sf1
        JOIN sep ON sf1.ticker = sep.ticker
        WHERE sf1.dimension = 'ARQ'
          AND sf1.datekey >= '2015-01-01'
        LIMIT 5
    """).df()
    print(date_compat.to_string(index=False))
    
    print("\nStep 3: Test join with explicit CAST...")
    cast_join = con.execute("""
        SELECT COUNT(*) as cnt
        FROM sf1 f
        JOIN sep p ON f.ticker = p.ticker 
            AND CAST(p.date AS DATE) >= CAST(f.datekey AS DATE)
            AND CAST(p.date AS DATE) < CAST(f.datekey AS DATE) + INTERVAL '95 days'
        WHERE f.dimension = 'ARQ'
          AND f.roe IS NOT NULL
          AND f.datekey >= '2015-01-01'
          AND p.date >= '2015-01-01'
        LIMIT 1000000
    """).fetchone()[0]
    print(f"  Join count with CAST: {cast_join:,}")
    
    print("\nStep 4: Check if ROE filter is too restrictive...")
    roe_check = con.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(roe) as has_roe,
            COUNT(CASE WHEN roe BETWEEN -1 AND 2 THEN 1 END) as roe_in_range,
            AVG(roe) as avg_roe,
            MIN(roe) as min_roe,
            MAX(roe) as max_roe
        FROM sf1
        WHERE dimension = 'ARQ'
          AND datekey >= '2015-01-01'
    """).df()
    print(roe_check.to_string(index=False))

    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    print("""
Based on this diagnostic, identify:

1. Are ticker codes matching between SF1 and SEP?
2. Are date formats compatible?
3. Is the date range filter too restrictive?
4. Are the value filters (ROE range, etc.) too restrictive?
5. Which feat_ tables have low coverage in feat_matrix?
""")

    con.close()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()