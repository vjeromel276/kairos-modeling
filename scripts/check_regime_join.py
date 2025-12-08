#!/usr/bin/env python3
"""Check the actual join in regime switching"""
import duckdb

con = duckdb.connect("data/kairos.duckdb", read_only=True)

print("Testing the join that regime_switching.py should be doing...")
print()

# Check if regime_history has a 'regime' column
cols = con.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'regime_history'
""").df()
print("Columns in regime_history:")
print(cols['column_name'].tolist())
print()

# Check backtest_results columns
cols_lo = con.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'backtest_results_longonly_r4'
""").df()
print("Columns in backtest_results_longonly_r4:")
print(cols_lo['column_name'].tolist())
print()

cols_ls = con.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'backtest_results_ls_opt'
""").df()
print("Columns in backtest_results_ls_opt:")
print(cols_ls['column_name'].tolist())
print()

# Try a manual join
print("Manual join test - regime + longonly:")
test = con.execute("""
    SELECT 
        r.date as regime_date,
        r.regime,
        lo.date as lo_date,
        lo.daily_ret as lo_ret
    FROM regime_history r
    JOIN backtest_results_longonly_r4 lo 
        ON CAST(r.date AS DATE) = CAST(lo.date AS DATE)
    WHERE r.date >= '2015-01-01'
    LIMIT 10
""").df()
print(test)
print()

# Check date ranges more carefully
print("Date range details:")
print()
ranges = con.execute("""
    SELECT 
        'regime_history' as tbl,
        MIN(date) as min_dt, 
        MAX(date) as max_dt,
        COUNT(*) as cnt
    FROM regime_history
    UNION ALL
    SELECT 
        'longonly_r4' as tbl,
        MIN(date), 
        MAX(date),
        COUNT(*)
    FROM backtest_results_longonly_r4
    UNION ALL
    SELECT 
        'ls_opt' as tbl,
        MIN(date), 
        MAX(date),
        COUNT(*)
    FROM backtest_results_ls_opt
""").df()
print(ranges)

con.close()