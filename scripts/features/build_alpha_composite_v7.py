#!/usr/bin/env python3
"""
build_alpha_composite_v7.py
===========================
Build alpha composite v7: 50/50 blend of v33_regime and quality_composite_z.

Based on empirical backtest optimization:
- v33_regime provides risk management and volatility timing
- quality_composite_z provides high-IC factor exposure
- 50/50 blend maximizes Active Sharpe (0.73) and total Sharpe (1.37)

Backtest Results (2015-2025):
| Blend (v33/quality) | Sharpe | Active Sharpe | Annual Return | Max DD |
|---------------------|--------|---------------|---------------|--------|
| 100/0 (v33 only)    | 1.32   | 0.54          | 26.36%        | -27.04%|
| 70/30               | 1.35   | 0.59          | 27.08%        | -23.86%|
| 60/40               | 1.37   | 0.65          | 27.46%        | -23.87%|
| **50/50** ✓         | 1.37   | 0.73          | 27.49%        | -24.30%|
| 40/60               | 1.18   | 0.51          | 23.52%        | -26.76%|

Requirements:
- feat_composite_v33_regime (risk management signal)
- feat_quality_v2 (quality factor from SF1 fundamentals)

Output table: feat_composite_v7

FIXED: Ensures date column is DATE type (not TIMESTAMP_NS) for proper joins.
"""

import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Build alpha composite v7 (50/50 blend)")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--v33-weight", type=float, default=0.50, help="Weight for v33_regime")
    parser.add_argument("--quality-weight", type=float, default=0.50, help="Weight for quality")
    args = parser.parse_args()

    # Normalize weights
    total = args.v33_weight + args.quality_weight
    w_v33 = args.v33_weight / total
    w_qual = args.quality_weight / total

    print(f"\n{'='*60}")
    print("BUILD ALPHA COMPOSITE V7")
    print(f"{'='*60}\n")
    print(f"Blend weights:")
    print(f"  v33_regime:        {w_v33:.0%}")
    print(f"  quality_composite: {w_qual:.0%}")

    con = duckdb.connect(args.db)

    # Check required tables
    print("\nChecking required tables...")
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()['table_name'].tolist()

    if 'feat_composite_v33_regime' not in tables:
        raise ValueError("Missing feat_composite_v33_regime - run the v33 build first")
    if 'feat_quality_v2' not in tables:
        raise ValueError("Missing feat_quality_v2 - run build_quality_factors_v2.py first")
    
    print("  ✓ feat_composite_v33_regime")
    print("  ✓ feat_quality_v2")

    # Build v7 with explicit DATE cast
    print("\nBuilding alpha_composite_v7...")
    
    con.execute("DROP TABLE IF EXISTS feat_composite_v7")
    con.execute(f"""
        CREATE TABLE feat_composite_v7 AS
        SELECT 
            a.ticker,
            CAST(a.date AS DATE) as date,
            {w_v33} * a.alpha_composite_v33_regime + {w_qual} * b.quality_composite_z as alpha_composite_v7
        FROM feat_composite_v33_regime a
        JOIN feat_quality_v2 b 
            ON a.ticker = b.ticker 
            AND CAST(a.date AS DATE) = CAST(b.date AS DATE)
        WHERE a.alpha_composite_v33_regime IS NOT NULL 
          AND b.quality_composite_z IS NOT NULL
    """)

    # Verify date type
    date_type = con.execute("SELECT typeof(date) FROM feat_composite_v7 LIMIT 1").fetchone()[0]
    print(f"  Output date type: {date_type}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_v7_ticker_date ON feat_composite_v7(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_v7_date ON feat_composite_v7(date)")

    # Stats
    stats = con.execute("""
        SELECT 
            COUNT(*) as n_rows,
            COUNT(DISTINCT ticker) as n_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(alpha_composite_v7) as avg_alpha,
            STDDEV(alpha_composite_v7) as std_alpha
        FROM feat_composite_v7
    """).fetchdf()
    
    n_rows = stats['n_rows'].iloc[0]
    print(f"\n✓ Created feat_composite_v7 with {n_rows:,} rows")
    print(f"\nSummary:")
    print(f"  Tickers: {stats['n_tickers'].iloc[0]:,}")
    print(f"  Date range: {stats['min_date'].iloc[0]} to {stats['max_date'].iloc[0]}")
    print(f"  Mean alpha: {stats['avg_alpha'].iloc[0]:.4f}")
    print(f"  Std alpha: {stats['std_alpha'].iloc[0]:.4f}")

    con.close()

    print(f"\n{'='*60}")
    print("DONE - Alpha composite v7 built successfully")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Rebuild feat_matrix_v2 to pick up the fixed v7")
    print("  2. Run backtest:")
    print("     python scripts/backtesting/backtest_academic_strategy_risk4.py \\")
    print("       --db data/kairos.duckdb \\")
    print("       --alpha-column alpha_composite_v7 \\")
    print("       --target-column ret_5d_f \\")
    print("       --top-n 75 --rebalance-every 5 --target-vol 0.20")
    print()


if __name__ == "__main__":
    main()