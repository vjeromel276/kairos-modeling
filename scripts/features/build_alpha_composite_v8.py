#!/usr/bin/env python3
"""
build_alpha_composite_v8.py
===========================
Build alpha composite v8: The best-performing alpha signal.

Formula: 35% v33_regime + 35% quality + 30% value

For dates where v33_regime is missing, falls back to:
  54% quality + 46% value (maintains the quality/value ratio)

Components:
- v33_regime: Risk management + volatility timing (IC=0.0133)
- quality_composite_z: ROE/ROA/Accruals (IC=0.0223)
- value_composite_z: Earnings/Book/EBITDA/Sales yields (IC=0.0216)

Key insight: Value is NEGATIVELY correlated with v7 (-0.05) but has high IC,
making it an ideal diversifier that adds return while reducing risk.

Backtest Results (2015-2025):
| Version | Formula                           | Sharpe | Active Sharpe | Annual | Max DD  |
|---------|-----------------------------------|--------|---------------|--------|---------|
| v33     | Original                          | 1.32   | 0.54          | 26.36% | -27.04% |
| v7      | 50% v33 + 50% quality             | 1.37   | 0.73          | 27.49% | -24.30% |
| v8      | 35% v33 + 35% quality + 30% value | 1.46   | 1.03          | 29.14% | -22.88% |

Improvement over v33: +10.6% Sharpe, +91% Active Sharpe

Requirements:
- feat_composite_v33_regime
- feat_quality_v2
- feat_value_v2

Output table: feat_composite_v8
"""

import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Build alpha composite v8")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    # Default weights based on optimization
    parser.add_argument("--v33-weight", type=float, default=0.35, help="Weight for v33_regime")
    parser.add_argument("--quality-weight", type=float, default=0.35, help="Weight for quality")
    parser.add_argument("--value-weight", type=float, default=0.30, help="Weight for value")
    parser.add_argument("--no-fallback", action="store_true", help="Don't use fallback for missing v33 dates")
    args = parser.parse_args()

    # Normalize weights
    total = args.v33_weight + args.quality_weight + args.value_weight
    w_v33 = args.v33_weight / total
    w_qual = args.quality_weight / total
    w_val = args.value_weight / total
    
    # Fallback weights when v33 is missing (maintain quality/value ratio)
    fallback_total = args.quality_weight + args.value_weight
    w_qual_fallback = args.quality_weight / fallback_total
    w_val_fallback = args.value_weight / fallback_total

    print(f"\n{'='*60}")
    print("BUILD ALPHA COMPOSITE V8 (BEST PERFORMER)")
    print(f"{'='*60}\n")
    print(f"Primary weights (when v33 available):")
    print(f"  v33_regime:        {w_v33:.0%} (risk management, IC=0.013)")
    print(f"  quality_composite: {w_qual:.0%} (ROE/ROA, IC=0.022)")
    print(f"  value_composite:   {w_val:.0%} (yields, IC=0.022)")
    print(f"\nFallback weights (when v33 missing):")
    print(f"  quality_composite: {w_qual_fallback:.0%}")
    print(f"  value_composite:   {w_val_fallback:.0%}")

    con = duckdb.connect(args.db)

    # Check required tables
    print("\nChecking required tables...")
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()['table_name'].tolist()

    required = ['feat_composite_v33_regime', 'feat_quality_v2', 'feat_value_v2']
    for t in required:
        if t not in tables:
            raise ValueError(f"Missing {t} - run the prerequisite build scripts first")
        print(f"  ✓ {t}")

    # Build v8 with fallback
    print("\nBuilding alpha_composite_v8...")
    
    con.execute("DROP TABLE IF EXISTS feat_composite_v8")
    
    if args.no_fallback:
        # Original behavior - only where all three exist
        con.execute(f"""
            CREATE TABLE feat_composite_v8 AS
            SELECT 
                v33.ticker,
                v33.date,
                {w_v33} * v33.alpha_composite_v33_regime 
                + {w_qual} * q.quality_composite_z 
                + {w_val} * v.value_composite_z as alpha_composite_v8
            FROM feat_composite_v33_regime v33
            JOIN feat_quality_v2 q ON v33.ticker = q.ticker AND v33.date = q.date
            JOIN feat_value_v2 v ON v33.ticker = v.ticker AND v33.date = v.date
            WHERE v33.alpha_composite_v33_regime IS NOT NULL 
              AND q.quality_composite_z IS NOT NULL
              AND v.value_composite_z IS NOT NULL
        """)
    else:
        # With fallback - use quality+value when v33 is missing
        con.execute(f"""
            CREATE TABLE feat_composite_v8 AS
            WITH base AS (
                -- All ticker-date combinations from quality and value
                SELECT 
                    q.ticker,
                    q.date,
                    q.quality_composite_z,
                    v.value_composite_z,
                    v33.alpha_composite_v33_regime
                FROM feat_quality_v2 q
                JOIN feat_value_v2 v ON q.ticker = v.ticker AND q.date = v.date
                LEFT JOIN feat_composite_v33_regime v33 ON q.ticker = v33.ticker AND q.date = v33.date
                WHERE q.quality_composite_z IS NOT NULL
                  AND v.value_composite_z IS NOT NULL
            )
            SELECT 
                ticker,
                date,
                CASE 
                    WHEN alpha_composite_v33_regime IS NOT NULL THEN
                        {w_v33} * alpha_composite_v33_regime 
                        + {w_qual} * quality_composite_z 
                        + {w_val} * value_composite_z
                    ELSE
                        {w_qual_fallback} * quality_composite_z 
                        + {w_val_fallback} * value_composite_z
                END as alpha_composite_v8
            FROM base
        """)

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_v8_ticker_date ON feat_composite_v8(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_v8_date ON feat_composite_v8(date)")

    # Stats
    stats = con.execute("""
        SELECT 
            COUNT(*) as n_rows,
            COUNT(DISTINCT ticker) as n_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(alpha_composite_v8) as avg_alpha,
            STDDEV(alpha_composite_v8) as std_alpha
        FROM feat_composite_v8
    """).fetchdf()
    
    n_rows = stats['n_rows'].iloc[0]
    print(f"\n✓ Created feat_composite_v8 with {n_rows:,} rows")
    print(f"\nSummary:")
    print(f"  Tickers: {stats['n_tickers'].iloc[0]:,}")
    print(f"  Date range: {stats['min_date'].iloc[0]} to {stats['max_date'].iloc[0]}")
    print(f"  Mean alpha: {stats['avg_alpha'].iloc[0]:.4f}")
    print(f"  Std alpha: {stats['std_alpha'].iloc[0]:.4f}")
    
    # Show coverage breakdown
    if not args.no_fallback:
        coverage = con.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN v33.alpha_composite_v33_regime IS NOT NULL THEN 1 ELSE 0 END) as with_v33,
                SUM(CASE WHEN v33.alpha_composite_v33_regime IS NULL THEN 1 ELSE 0 END) as fallback_only
            FROM feat_composite_v8 v8
            LEFT JOIN feat_composite_v33_regime v33 ON v8.ticker = v33.ticker AND v8.date = v33.date
        """).fetchdf()
        total = coverage['total'].iloc[0]
        with_v33 = coverage['with_v33'].iloc[0]
        fallback = coverage['fallback_only'].iloc[0]
        print(f"\nCoverage breakdown:")
        print(f"  With v33 (full formula): {with_v33:,} ({100*with_v33/total:.1f}%)")
        print(f"  Fallback (quality+value): {fallback:,} ({100*fallback/total:.1f}%)")

    # Update feat_matrix_v2 if it exists
    if 'feat_matrix_v2' in tables:
        print("\nUpdating feat_matrix_v2...")
        
        # Check if column exists
        cols = con.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'feat_matrix_v2'
        """).fetchdf()['column_name'].tolist()
        
        if 'alpha_composite_v8' not in cols:
            con.execute("ALTER TABLE feat_matrix_v2 ADD COLUMN alpha_composite_v8 DOUBLE")
        
        con.execute(f"""
            UPDATE feat_matrix_v2 m
            SET alpha_composite_v8 = v8.alpha_composite_v8
            FROM feat_composite_v8 v8
            WHERE m.ticker = v8.ticker AND m.date = v8.date
        """)
        
        coverage = con.execute("""
            SELECT AVG(CASE WHEN alpha_composite_v8 IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100 as pct
            FROM feat_matrix_v2
        """).fetchone()[0]
        print(f"  ✓ Updated feat_matrix_v2 (coverage: {coverage:.1f}%)")

    con.close()

    print(f"\n{'='*60}")
    print("DONE - Alpha composite v8 built successfully")
    print(f"{'='*60}")
    print("\nExpected Performance (2015-2025 backtest):")
    print("  Sharpe:        1.46")
    print("  Active Sharpe: 1.03")
    print("  Annual Return: 29.14%")
    print("  Max Drawdown:  -22.88%")
    print("\nBacktest command:")
    print("  python scripts/backtesting/backtest_academic_strategy_risk4.py \\")
    print("    --db data/kairos.duckdb \\")
    print("    --alpha-column alpha_composite_v8 \\")
    print("    --target-column ret_5d_f \\")
    print("    --top-n 75 --rebalance-every 5 --target-vol 0.20")
    print()


if __name__ == "__main__":
    main()