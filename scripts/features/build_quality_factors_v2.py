#!/usr/bin/env python3
"""
build_quality_factors_v2.py
===========================
Build quality factors by computing ROE/ROA/Accruals from raw SF1 data.

Based on Phase 4 Investigation findings:
- SF1.roe, SF1.roa, SF1.roic columns are EMPTY (0% coverage)
- BUT we can compute them from raw components:
  - netinc: 96.6% coverage
  - equity: 99.9% coverage  
  - assets: 99.9% coverage
  - ncfo: 94.8% coverage (operating cash flow for accruals)

Computed factors:
- ROE = netinc / equity
- ROA = netinc / assets
- Accruals = (netinc - ncfo) / assets  (lower is better - negative IC expected)
- Quality composite = blend of above

Output table: feat_quality_v2
Columns: ticker, date, computed_roe, computed_roa, computed_accruals, quality_composite_z

NOTE: Column names use 'computed_' prefix to avoid conflicts with empty SF1 columns.
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
    parser = argparse.ArgumentParser(description="Build quality factors v2 from SF1 raw data")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--dimension", default="ARQ", help="SF1 dimension to use (ARQ, ARY, MRQ, etc.)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BUILD QUALITY FACTORS V2 (Computed from SF1 Raw)")
    print(f"{'='*60}\n")

    con = duckdb.connect(args.db)

    # Check SF1 table and required columns
    print("Checking sf1 table...")
    
    schema = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'sf1'
    """).fetchall()
    cols = [c[0] for c in schema]
    print(f"  SF1 has {len(cols)} columns")
    
    # Check for required columns - try alternative names
    required_base = ['ticker', 'netinc', 'equity', 'assets']
    date_col = 'datekey' if 'datekey' in cols else 'calendardate' if 'calendardate' in cols else 'date'
    
    missing = [c for c in required_base if c not in cols]
    if missing:
        print(f"  ⚠ Missing columns: {missing}")
        print(f"  Available columns: {cols[:20]}...")
        raise ValueError(f"Missing required columns in sf1: {missing}")
    
    print(f"  ✓ Required columns present")
    print(f"  Using date column: {date_col}")

    # Check for ncfo (optional - for accruals)
    has_ncfo = 'ncfo' in cols
    if has_ncfo:
        print(f"  ✓ ncfo available for accruals calculation")
    else:
        print(f"  ⚠ ncfo not available - skipping accruals")

    # Check dimension column
    has_dimension = 'dimension' in cols
    if has_dimension:
        dims = con.execute("SELECT DISTINCT dimension FROM sf1").fetchdf()
        print(f"  Available dimensions: {dims['dimension'].tolist()}")
    else:
        print(f"  No dimension column - using all data")
    
    # Get row counts
    count = con.execute("SELECT COUNT(*) FROM sf1").fetchone()[0]
    print(f"  Total rows in sf1: {count:,}")

    # Build quality factors
    print(f"\nComputing quality factors...")
    
    # Build query based on available columns
    dim_filter = f"AND dimension = '{args.dimension}'" if has_dimension else ""
    ncfo_select = "ncfo," if has_ncfo else "NULL as ncfo,"
    accruals_calc = """
        CASE 
            WHEN assets > 0 AND ncfo IS NOT NULL 
            THEN (netinc - ncfo) / assets 
            ELSE NULL 
        END AS computed_accruals""" if has_ncfo else "NULL AS computed_accruals"
    
    query = f"""
    SELECT 
        ticker,
        {date_col} as date,
        netinc,
        equity,
        assets,
        {ncfo_select}
        -- Compute ROE (handle division by zero and extreme values)
        CASE 
            WHEN equity > 0 AND ABS(netinc / equity) < 2.0 
            THEN netinc / equity 
            ELSE NULL 
        END AS computed_roe,
        -- Compute ROA
        CASE 
            WHEN assets > 0 AND ABS(netinc / assets) < 0.5 
            THEN netinc / assets 
            ELSE NULL 
        END AS computed_roa,
        -- Compute Accruals (earnings - cash flow) / assets
        -- Lower accruals = higher quality earnings
        {accruals_calc}
    FROM sf1
    WHERE {date_col} >= '2000-01-01'
      AND netinc IS NOT NULL
      AND (equity > 0 OR assets > 0)
      {dim_filter}
    """
    
    print("  Executing query...")
    try:
        df = con.execute(query).fetchdf()
        print(f"  Loaded {len(df):,} rows")
    except Exception as e:
        print(f"  ⚠ Query error: {e}")
        print("  Trying simplified query...")
        # Fallback to simpler query
        query = f"""
        SELECT 
            ticker,
            {date_col} as date,
            CASE WHEN equity > 0 THEN netinc / equity ELSE NULL END AS computed_roe,
            CASE WHEN assets > 0 THEN netinc / assets ELSE NULL END AS computed_roa,
            NULL AS computed_accruals
        FROM sf1
        WHERE {date_col} >= '2000-01-01'
          AND netinc IS NOT NULL
        """
        df = con.execute(query).fetchdf()
        print(f"  Loaded {len(df):,} rows (simplified)")

    if len(df) == 0:
        print("  ⚠ No data loaded - check SF1 table structure")
        con.close()
        return

    # Coverage stats
    for col in ['computed_roe', 'computed_roa', 'computed_accruals']:
        if col in df.columns:
            coverage = df[col].notna().mean() * 100
            print(f"  {col}: {coverage:.1f}% coverage")

    # Additional filtering for extreme values
    print("\nFiltering extreme values...")
    df['computed_roe'] = df['computed_roe'].clip(-1.0, 1.0)  # -100% to +100%
    df['computed_roa'] = df['computed_roa'].clip(-0.3, 0.3)  # -30% to +30%
    if 'computed_accruals' in df.columns:
        df['computed_accruals'] = df['computed_accruals'].clip(-1.0, 1.0)
    else:
        df['computed_accruals'] = np.nan

    # Forward fill to daily frequency
    # SF1 data is quarterly, so we need to expand to daily
    print("\nExpanding to daily frequency...")
    
    # Get trading dates from SEP
    trading_dates = con.execute("""
        SELECT DISTINCT date FROM sep 
        WHERE date >= '2000-01-01'
        ORDER BY date
    """).fetchdf()
    
    # For each ticker, forward-fill to daily
    results = []
    tickers = df['ticker'].unique()
    quality_cols = ['computed_roe', 'computed_roa', 'computed_accruals']
    
    for i, ticker in enumerate(tickers):
        if i % 500 == 0:
            print(f"  Processing {i:,} / {len(tickers):,} tickers")
        
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        # Get date range for this ticker
        min_date = ticker_data['date'].min()
        max_date = ticker_data['date'].max()
        
        # Filter trading dates
        ticker_dates = trading_dates[
            (trading_dates['date'] >= min_date) & 
            (trading_dates['date'] <= max_date)
        ].copy()
        ticker_dates['ticker'] = ticker
        
        # Merge and forward fill
        merged = ticker_dates.merge(
            ticker_data[['ticker', 'date'] + quality_cols], 
            on=['ticker', 'date'], 
            how='left'
        )
        merged = merged.sort_values('date')
        merged[quality_cols] = merged[quality_cols].ffill()
        
        results.append(merged)

    print(f"  Processed {len(tickers):,} tickers")

    # Combine results
    print("\nCombining results...")
    df = pd.concat(results, ignore_index=True)
    print(f"  Total rows: {len(df):,}")

    # Drop rows with no quality data
    df = df.dropna(subset=['computed_roe', 'computed_roa'], how='all')
    print(f"  Rows after dropna: {len(df):,}")

    # Compute cross-sectional z-scores by date
    print("Computing cross-sectional z-scores...")
    
    def zscore_group(group):
        group['roe_z'] = zscore_winsorize(group['computed_roe'])
        group['roa_z'] = zscore_winsorize(group['computed_roa'])
        # For accruals, lower is better, so negate
        if group['computed_accruals'].notna().any():
            group['accruals_z'] = -1 * zscore_winsorize(group['computed_accruals'])
        else:
            group['accruals_z'] = 0.0
        return group
    
    df = df.groupby('date', group_keys=False).apply(zscore_group)
    
    # Quality composite: equal weight ROE, ROA, negative accruals
    print("Computing quality composite...")
    df['quality_composite_z'] = df[['roe_z', 'roa_z', 'accruals_z']].mean(axis=1, skipna=True)

    # Select final columns
    final_cols = ['ticker', 'date', 'computed_roe', 'computed_roa', 'computed_accruals', 'quality_composite_z']
    df = df[final_cols].copy()

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_quality_v2")
    con.execute("""
        CREATE TABLE feat_quality_v2 (
            ticker VARCHAR,
            date DATE,
            computed_roe DOUBLE,
            computed_roa DOUBLE,
            computed_accruals DOUBLE,
            quality_composite_z DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_quality_v2 SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_quality_v2_ticker_date ON feat_quality_v2(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_quality_v2_date ON feat_quality_v2(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_quality_v2").fetchone()[0]
    print(f"\n✓ Created feat_quality_v2 with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(quality_composite_z) as avg_composite,
            STDDEV(quality_composite_z) as std_composite,
            AVG(computed_roe) as avg_roe,
            AVG(computed_roa) as avg_roa
        FROM feat_quality_v2
    """).fetchdf()
    print(stats.to_string())

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Quality factors v2 built successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()