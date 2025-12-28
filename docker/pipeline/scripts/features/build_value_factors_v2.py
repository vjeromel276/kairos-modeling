#!/usr/bin/env python3
"""
build_value_factors_v2.py
=========================
Build institutional-quality value factors from the Sharadar 'daily' table.

Based on Phase 4 Investigation findings:
- EBITDA yield: IC +0.0173 (strongest)
- Book yield:   IC +0.0142
- Earnings yield: IC +0.0141
- Sales yield: IC ~0.01

Key insight: D10 (cheapest) stocks are value traps with only 4% annual return.
D9 (cheap but not distressed) has 50% annual return.
Solution: Quality-adjusted value that penalizes extreme cheapness.

Output table: feat_value_v2
Columns: ticker, date, earnings_yield, book_yield, ebitda_yield, sales_yield,
         value_composite_z, value_quality_adj
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

def zscore_winsorize(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score with winsorization at +/- clip."""
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    z = (s - mean) / std
    return z.clip(-clip, clip)

def main():
    parser = argparse.ArgumentParser(description="Build value factors v2 from daily table")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BUILD VALUE FACTORS V2 (From Sharadar Daily)")
    print(f"{'='*60}\n")

    con = duckdb.connect(args.db)

    # Check daily table exists and has required columns
    print("Checking daily table...")
    schema = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'daily'
    """).fetchall()
    cols = [c[0] for c in schema]
    print(f"  Found {len(cols)} columns in daily table")

    required = ['ticker', 'date', 'pe', 'pb', 'ps', 'evebitda']
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns in daily: {missing}")
    print(f"  ✓ All required columns present")

    # Get row count
    count = con.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    print(f"  Total rows in daily: {count:,}")

    # Build value factors with proper filtering
    print("\nBuilding value factors...")
    
    query = """
    SELECT 
        ticker,
        date,
        -- Raw yields (inverse of ratios, filtered for reasonable values)
        CASE 
            WHEN pe > 0 AND pe < 100 THEN 1.0 / pe 
            ELSE NULL 
        END AS earnings_yield,
        CASE 
            WHEN pb > 0 AND pb < 20 THEN 1.0 / pb 
            ELSE NULL 
        END AS book_yield,
        CASE 
            WHEN evebitda > 0 AND evebitda < 50 THEN 1.0 / evebitda 
            ELSE NULL 
        END AS ebitda_yield,
        CASE 
            WHEN ps > 0 AND ps < 20 THEN 1.0 / ps 
            ELSE NULL 
        END AS sales_yield
    FROM daily
    WHERE date >= '2000-01-01'
    """
    
    print("  Executing query (this may take a few minutes)...")
    df = con.execute(query).fetchdf()
    print(f"  Loaded {len(df):,} rows")

    # Coverage stats
    for col in ['earnings_yield', 'book_yield', 'ebitda_yield', 'sales_yield']:
        coverage = df[col].notna().mean() * 100
        print(f"  {col}: {coverage:.1f}% coverage")

    # Compute cross-sectional z-scores by date
    print("\nComputing cross-sectional z-scores...")
    
    yield_cols = ['earnings_yield', 'book_yield', 'ebitda_yield', 'sales_yield']
    z_cols = [f'{c}_z' for c in yield_cols]
    
    # Group by date and z-score
    def zscore_group(group):
        for col, zcol in zip(yield_cols, z_cols):
            group[zcol] = zscore_winsorize(group[col])
        return group
    
    df = df.groupby('date', group_keys=False).apply(zscore_group)
    
    # Composite: equal-weight average of available z-scores
    print("Computing composite value score...")
    df['value_composite_z'] = df[z_cols].mean(axis=1, skipna=True)
    
    # Quality-adjusted value: penalize extreme values (D10 avoidance)
    # If a stock is in top 10% cheapest (highest yield), apply penalty
    print("Computing quality-adjusted value (D10 penalty)...")
    
    def adjust_for_traps(group):
        # For each yield, penalize if in top decile
        composite = group['value_composite_z'].copy()
        
        for col in yield_cols:
            if group[col].notna().sum() > 100:  # Need enough data
                d10_threshold = group[col].quantile(0.90)
                is_d10 = group[col] >= d10_threshold
                # Penalty: reduce score by 1 z-score unit for D10 stocks
                composite = composite.where(~is_d10, composite - 1.0)
        
        group['value_quality_adj'] = composite
        return group
    
    df = df.groupby('date', group_keys=False).apply(adjust_for_traps)
    
    # Select final columns
    final_cols = ['ticker', 'date', 'earnings_yield', 'book_yield', 'ebitda_yield', 
                  'sales_yield', 'value_composite_z', 'value_quality_adj']
    df = df[final_cols].copy()
    
    # Drop rows with no value data
    df = df.dropna(subset=['value_composite_z'])
    print(f"  Final rows after dropna: {len(df):,}")

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_value_v2")
    con.execute("""
        CREATE TABLE feat_value_v2 (
            ticker VARCHAR,
            date DATE,
            earnings_yield DOUBLE,
            book_yield DOUBLE,
            ebitda_yield DOUBLE,
            sales_yield DOUBLE,
            value_composite_z DOUBLE,
            value_quality_adj DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_value_v2 SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_value_v2_ticker_date ON feat_value_v2(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_value_v2_date ON feat_value_v2(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_value_v2").fetchone()[0]
    print(f"\n✓ Created feat_value_v2 with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(value_composite_z) as avg_composite,
            STDDEV(value_composite_z) as std_composite
        FROM feat_value_v2
    """).fetchdf()
    print(stats.to_string())

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Value factors v2 built successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
