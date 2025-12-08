#!/usr/bin/env python3
"""
build_momentum_factors_v2.py
============================
Build institutional-quality momentum factors from the Sharadar 'sep' table.

Based on Phase 4 Investigation findings:
- 12-month momentum: IC +0.0123
- 12-1 month momentum: IC +0.0118 (skip recent month - best)
- 1-month reversal: IC +0.0077 (mean reversion)
- 3-month momentum: IC -0.0078 (reversal, not momentum)
- 6-month momentum: IC -0.0002 (neutral)

Key insight: Short-term (<6m) shows reversal, long-term (12m) shows momentum.
Skip the most recent month for 12m momentum (avoids short-term reversal).

Output table: feat_momentum_v2
Columns: ticker, date, mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1, 
         reversal_1m, momentum_composite_z
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
    parser = argparse.ArgumentParser(description="Build momentum factors v2 from SEP")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BUILD MOMENTUM FACTORS V2 (From Sharadar SEP)")
    print(f"{'='*60}\n")

    con = duckdb.connect(args.db)

    # Check SEP table
    print("Checking sep table...")
    count = con.execute("SELECT COUNT(*) FROM sep").fetchone()[0]
    print(f"  Total rows in sep: {count:,}")

    # Build momentum factors using window functions
    # This is more efficient than pandas groupby for large datasets
    print("\nBuilding momentum factors via SQL...")
    
    query = """
    WITH price_data AS (
        SELECT 
            ticker,
            date,
            closeadj,
            -- Lagged prices for momentum calculation
            LAG(closeadj, 21) OVER (PARTITION BY ticker ORDER BY date) AS price_1m_ago,
            LAG(closeadj, 63) OVER (PARTITION BY ticker ORDER BY date) AS price_3m_ago,
            LAG(closeadj, 126) OVER (PARTITION BY ticker ORDER BY date) AS price_6m_ago,
            LAG(closeadj, 252) OVER (PARTITION BY ticker ORDER BY date) AS price_12m_ago,
            LAG(closeadj, 231) OVER (PARTITION BY ticker ORDER BY date) AS price_11m_ago  -- 252 - 21 = 231
        FROM sep
        WHERE closeadj > 0 
          AND date >= '1999-01-01'  -- Need 1 year of history for 12m momentum starting 2000
    )
    SELECT 
        ticker,
        date,
        -- Raw momentum returns
        (closeadj / NULLIF(price_1m_ago, 0)) - 1 AS mom_1m,
        (closeadj / NULLIF(price_3m_ago, 0)) - 1 AS mom_3m,
        (closeadj / NULLIF(price_6m_ago, 0)) - 1 AS mom_6m,
        (closeadj / NULLIF(price_12m_ago, 0)) - 1 AS mom_12m,
        -- 12-1 month momentum: return from 12m ago to 1m ago (skip recent month)
        (price_1m_ago / NULLIF(price_12m_ago, 0)) - 1 AS mom_12_1,
        -- 1-month reversal (negative of 1m momentum)
        -1 * ((closeadj / NULLIF(price_1m_ago, 0)) - 1) AS reversal_1m
    FROM price_data
    WHERE date >= '2000-01-01'
      AND price_12m_ago IS NOT NULL
    """
    
    print("  Executing query (this may take a few minutes)...")
    df = con.execute(query).fetchdf()
    print(f"  Loaded {len(df):,} rows")

    # Coverage stats
    mom_cols = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1', 'reversal_1m']
    for col in mom_cols:
        coverage = df[col].notna().mean() * 100
        print(f"  {col}: {coverage:.1f}% coverage")

    # Filter out extreme returns (data errors or delistings)
    print("\nFiltering extreme returns...")
    for col in mom_cols:
        # Cap at +/- 500% return
        df[col] = df[col].clip(-5.0, 5.0)
        # Set returns > 300% to NaN (likely errors)
        df.loc[df[col].abs() > 3.0, col] = np.nan

    # Compute cross-sectional z-scores by date
    print("Computing cross-sectional z-scores...")
    
    z_cols = [f'{c}_z' for c in mom_cols]
    
    def zscore_group(group):
        for col, zcol in zip(mom_cols, z_cols):
            group[zcol] = zscore_winsorize(group[col])
        return group
    
    df = df.groupby('date', group_keys=False).apply(zscore_group)
    
    # Momentum composite: weighted average of mom_12_1 and reversal_1m
    # Weights based on IC: mom_12_1 has IC 0.0118, reversal_1m has IC 0.0077
    # IC-weighted: 0.0118/(0.0118+0.0077) = 0.605, 0.0077/(0.0118+0.0077) = 0.395
    print("Computing momentum composite...")
    w_mom = 0.60
    w_rev = 0.40
    df['momentum_composite_z'] = (
        w_mom * df['mom_12_1_z'].fillna(0) + 
        w_rev * df['reversal_1m_z'].fillna(0)
    )
    
    # Select final columns
    final_cols = ['ticker', 'date', 'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 
                  'mom_12_1', 'reversal_1m', 'momentum_composite_z']
    df = df[final_cols].copy()
    
    # Drop rows with no momentum data
    df = df.dropna(subset=['mom_12_1'])
    print(f"  Final rows after dropna: {len(df):,}")

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_momentum_v2")
    con.execute("""
        CREATE TABLE feat_momentum_v2 (
            ticker VARCHAR,
            date DATE,
            mom_1m DOUBLE,
            mom_3m DOUBLE,
            mom_6m DOUBLE,
            mom_12m DOUBLE,
            mom_12_1 DOUBLE,
            reversal_1m DOUBLE,
            momentum_composite_z DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_momentum_v2 SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_momentum_v2_ticker_date ON feat_momentum_v2(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_momentum_v2_date ON feat_momentum_v2(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_momentum_v2").fetchone()[0]
    print(f"\n✓ Created feat_momentum_v2 with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(momentum_composite_z) as avg_composite,
            STDDEV(momentum_composite_z) as std_composite
        FROM feat_momentum_v2
    """).fetchdf()
    print(stats.to_string())

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Momentum factors v2 built successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
