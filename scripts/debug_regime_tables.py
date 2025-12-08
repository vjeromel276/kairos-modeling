#!/usr/bin/env python3
"""
Debug regime switching tables - check date formats and ranges
"""

import argparse
import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 70)
    print("REGIME SWITCHING TABLE DIAGNOSTICS")
    print("=" * 70)

    # 1. Check what tables exist
    print("\n1. RELEVANT TABLES IN DATABASE")
    print("-" * 50)
    tables = con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE '%regime%' 
           OR table_name LIKE '%backtest%'
        ORDER BY table_name
    """).df()
    print(tables.to_string(index=False))

    # 2. Check regime_history table
    print("\n\n2. REGIME HISTORY TABLE")
    print("-" * 50)
    try:
        regime_info = con.execute("""
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                typeof(date) as date_type
            FROM regime_history
        """).df()
        print(regime_info.to_string(index=False))
        
        print("\nSample rows:")
        sample = con.execute("SELECT * FROM regime_history LIMIT 5").df()
        print(sample.to_string(index=False))
        
        print("\nRegime distribution:")
        dist = con.execute("""
            SELECT regime, COUNT(*) as cnt 
            FROM regime_history 
            GROUP BY regime 
            ORDER BY regime
        """).df()
        print(dist.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    # 3. Check backtest_results_longonly_r4
    print("\n\n3. BACKTEST_RESULTS_LONGONLY_R4 TABLE")
    print("-" * 50)
    try:
        lo_info = con.execute("""
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                typeof(date) as date_type
            FROM backtest_results_longonly_r4
        """).df()
        print(lo_info.to_string(index=False))
        
        print("\nSample rows:")
        sample = con.execute("SELECT * FROM backtest_results_longonly_r4 LIMIT 5").df()
        print(sample.to_string(index=False))
        
        print("\nColumns:")
        cols = con.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'backtest_results_longonly_r4'
        """).df()
        print(cols.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    # 4. Check backtest_results_ls_opt
    print("\n\n4. BACKTEST_RESULTS_LS_OPT TABLE")
    print("-" * 50)
    try:
        ls_info = con.execute("""
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                typeof(date) as date_type
            FROM backtest_results_ls_opt
        """).df()
        print(ls_info.to_string(index=False))
        
        print("\nSample rows:")
        sample = con.execute("SELECT * FROM backtest_results_ls_opt LIMIT 5").df()
        print(sample.to_string(index=False))
        
        print("\nColumns:")
        cols = con.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'backtest_results_ls_opt'
        """).df()
        print(cols.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    # 5. Check for date overlap
    print("\n\n5. DATE OVERLAP CHECK")
    print("-" * 50)
    try:
        overlap = con.execute("""
            SELECT 
                (SELECT COUNT(DISTINCT date) FROM regime_history) as regime_dates,
                (SELECT COUNT(DISTINCT date) FROM backtest_results_longonly_r4) as lo_dates,
                (SELECT COUNT(DISTINCT date) FROM backtest_results_ls_opt) as ls_dates
        """).df()
        print(overlap.to_string(index=False))
        
        # Try to find overlapping dates with explicit cast
        print("\nTrying to find overlaps with CAST...")
        overlap_test = con.execute("""
            SELECT COUNT(*) as overlap_count
            FROM regime_history r
            JOIN backtest_results_longonly_r4 lo 
                ON CAST(r.date AS DATE) = CAST(lo.date AS DATE)
        """).df()
        print(f"Regime ∩ Long-only: {overlap_test['overlap_count'].values[0]} dates")
        
        overlap_test2 = con.execute("""
            SELECT COUNT(*) as overlap_count
            FROM regime_history r
            JOIN backtest_results_ls_opt ls 
                ON CAST(r.date AS DATE) = CAST(ls.date AS DATE)
        """).df()
        print(f"Regime ∩ LS-opt: {overlap_test2['overlap_count'].values[0]} dates")
        
    except Exception as e:
        print(f"Error: {e}")

    # 6. Show actual date values to compare formats
    print("\n\n6. RAW DATE VALUE COMPARISON")
    print("-" * 50)
    try:
        print("Regime dates (first 3):")
        rd = con.execute("SELECT date FROM regime_history ORDER BY date LIMIT 3").df()
        for d in rd['date']:
            print(f"  {d} (type: {type(d)})")
        
        print("\nLong-only dates (first 3):")
        ld = con.execute("SELECT date FROM backtest_results_longonly_r4 ORDER BY date LIMIT 3").df()
        for d in ld['date']:
            print(f"  {d} (type: {type(d)})")
            
        print("\nLS-opt dates (first 3):")
        lsd = con.execute("SELECT date FROM backtest_results_ls_opt ORDER BY date LIMIT 3").df()
        for d in lsd['date']:
            print(f"  {d} (type: {type(d)})")
    except Exception as e:
        print(f"Error: {e}")

    con.close()
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()