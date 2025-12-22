#!/usr/bin/env python3
"""
build_feature_matrix_v2.py
==========================
Build complete feature matrix including both legacy features and new v2 factors.

FIXED: Handles duplicate columns and missing data gracefully.

This script:
1. Loads all feat_* tables
2. Joins them on ticker/date (with duplicate column handling)
3. Adds the new v2 factors (value, momentum, quality, insider, composite_v5)
4. Creates the final feat_matrix_v2 table
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

def safe_merge(base: pd.DataFrame, new_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Merge with duplicate column handling."""
    # Find overlapping columns (excluding join keys)
    join_cols = ['ticker', 'date']
    base_cols = set(base.columns) - set(join_cols)
    new_cols = set(new_df.columns) - set(join_cols)
    overlap = base_cols & new_cols
    
    if overlap:
        # Rename overlapping columns in new_df with table suffix
        rename_map = {col: f"{col}_{table_name}" for col in overlap}
        new_df = new_df.rename(columns=rename_map)
        print(f"    Renamed {len(overlap)} duplicate cols: {list(overlap)[:3]}...")
    
    return base.merge(new_df, on=join_cols, how='left')


def main():
    parser = argparse.ArgumentParser(description="Build feature matrix v2")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--date", required=True, help="Latest date to include (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Path to universe CSV")
    parser.add_argument("--min-date", default="2010-01-01", help="Minimum date")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("BUILD FEATURE MATRIX V2 (FIXED)")
    print(f"{'='*70}\n")
    print(f"Date range: {args.min_date} to {args.date}")

    con = duckdb.connect(args.db)

    # Load universe
    print("\nLoading universe...")
    universe = pd.read_csv(args.universe)
    tickers = universe['ticker'].unique().tolist()
    print(f"  Universe: {len(tickers):,} tickers")

    # Get all feat_ tables
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main' AND table_name LIKE 'feat_%'
    """).fetchdf()['table_name'].tolist()
    
    print(f"\nAvailable feature tables: {len(tables)}")

    # Define priority tables (order matters - later tables override earlier for duplicates)
    # V2 tables should come last to take precedence
    priority_tables = [
        # Core features (high coverage, stable)
        'feat_price_action',
        'feat_price_shape', 
        'feat_stat',
        'feat_trend',
        'feat_volume_volatility',
        'feat_targets',
        'feat_adv',  # ADV for liquidity filtering
        
        # Intermediate composites
        'feat_composite_academic',
        'feat_institutional_academic',
        'feat_composite_v31',
        'feat_vol_sizing',
        'feat_beta',
        
        # V2 factors (highest priority)
        'feat_value_v2',
        'feat_momentum_v2', 
        'feat_quality_v2',
        'feat_insider',
        'feat_composite_v5',
        'feat_composite_v6',  # IC-corrected weights
        'feat_composite_v7',  # 50/50 blend - BEST
        'feat_composite_v32b',   # new CS+CL v2 blend
        'feat_composite_v8',        # still your CS+CL v1 blend for now
    ]
    
    # Only include tables that exist
    tables_to_use = [t for t in priority_tables if t in tables]
    
    # Also try to get v33_regime directly if it exists
    if 'feat_composite_v33_regime' in tables:
        tables_to_use.append('feat_composite_v33_regime')
    
    print(f"Tables to include: {len(tables_to_use)}")
    for t in tables_to_use:
        print(f"  - {t}")

    # Start with base dates from SEP
    print("\nBuilding base grid from SEP...")
    
    base_query = f"""
        SELECT DISTINCT ticker, date
        FROM sep
        WHERE date BETWEEN '{args.min_date}' AND '{args.date}'
        ORDER BY ticker, date
    """
    base = con.execute(base_query).fetchdf()
    
    # Filter to universe
    base = base[base['ticker'].isin(tickers)]
    print(f"  Base grid: {len(base):,} ticker-date pairs")

    # Join each table
    print("\nJoining feature tables...")
    
    for table in tables_to_use:
        try:
            # Get columns from table schema
            cols_df = con.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).fetchdf()
            all_cols = cols_df['column_name'].tolist()
            
            # Feature columns (excluding join keys)
            feat_cols = [c for c in all_cols if c not in ['ticker', 'date']]
            
            if not feat_cols:
                print(f"  ⚠ {table}: no feature columns, skipping")
                continue
            
            # Build select with explicit column list
            select_cols = ['ticker', 'date'] + feat_cols
            select_str = ', '.join([f'"{c}"' for c in select_cols])
            
            # Load data
            query = f"""
                SELECT {select_str}
                FROM {table}
                WHERE date BETWEEN '{args.min_date}' AND '{args.date}'
            """
            df = con.execute(query).fetchdf()
            
            if len(df) == 0:
                print(f"  ⚠ {table}: no data in date range, skipping")
                continue
            
            # Safe merge with duplicate handling
            base = safe_merge(base, df, table.replace('feat_', ''))
            
            # Check coverage of first feature column
            first_col = feat_cols[0]
            # Find the actual column name (might have been renamed)
            actual_col = first_col if first_col in base.columns else f"{first_col}_{table.replace('feat_', '')}"
            if actual_col in base.columns:
                coverage = base[actual_col].notna().mean() * 100
            else:
                coverage = 0.0
            
            print(f"  ✓ {table}: {len(feat_cols)} cols, {coverage:.1f}% coverage")
            
        except Exception as e:
            print(f"  ⚠ {table}: error - {str(e)[:50]}")

    print(f"\nFinal matrix: {len(base):,} rows, {len(base.columns)} columns")

    # Identify key alpha columns
    alpha_cols = [c for c in base.columns if 'alpha' in c.lower() or 'composite' in c.lower()]
    print(f"\nAlpha/composite columns found: {len(alpha_cols)}")
    for col in sorted(alpha_cols):
        coverage = base[col].notna().mean() * 100
        print(f"  {col}: {coverage:.1f}%")

    # Ensure we have the key columns for backtesting
    required_for_backtest = ['alpha_composite_v5', 'alpha_composite_v6', 'alpha_composite_v7', 'ret_5d_f']
    missing_required = [c for c in required_for_backtest if c not in base.columns]
    if missing_required:
        print(f"\n⚠ WARNING: Missing required columns for backtest: {missing_required}")

    # Identify and handle non-numeric columns
    print("\nChecking column types...")
    string_cols = []
    numeric_cols = []
    for col in base.columns:
        if col in ['ticker', 'date']:
            continue
        # Check if column has string data
        if base[col].dtype == 'object' or (base[col].dropna().apply(lambda x: isinstance(x, str)).any() if len(base[col].dropna()) > 0 else False):
            string_cols.append(col)
        else:
            numeric_cols.append(col)
    
    if string_cols:
        print(f"  Found {len(string_cols)} string columns: {string_cols}")
        print(f"  Converting to numeric or dropping...")
        for col in string_cols:
            # For regime columns, we can encode them or drop them
            if 'regime' in col.lower():
                # Encode regime as numeric
                unique_vals = base[col].dropna().unique()
                regime_map = {v: i for i, v in enumerate(unique_vals)}
                base[f'{col}_encoded'] = base[col].map(regime_map)
                print(f"    Encoded {col} -> {col}_encoded ({len(unique_vals)} unique values)")
            # Drop the original string column
            base = base.drop(columns=[col])
            print(f"    Dropped string column: {col}")

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_matrix_v2")
    
    # Create table with proper types
    col_defs = ['ticker VARCHAR', 'date DATE']
    for col in base.columns:
        if col not in ['ticker', 'date']:
            # Escape column names that might have special characters
            safe_col = col.replace('"', '""')
            col_defs.append(f'"{safe_col}" DOUBLE')
    
    create_sql = f"CREATE TABLE feat_matrix_v2 ({', '.join(col_defs)})"
    con.execute(create_sql)
    
    # Insert in batches
    batch_size = 100000
    total_inserted = 0
    for i in range(0, len(base), batch_size):
        batch = base.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_matrix_v2 SELECT * FROM batch")
        total_inserted = min(i+batch_size, len(base))
        if total_inserted % 500000 == 0 or total_inserted == len(base):
            print(f"  Inserted {total_inserted:,} / {len(base):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_matrix_v2_ticker_date ON feat_matrix_v2(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_matrix_v2_date ON feat_matrix_v2(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_matrix_v2").fetchone()[0]
    n_cols = con.execute("""
        SELECT COUNT(*) FROM information_schema.columns 
        WHERE table_name = 'feat_matrix_v2'
    """).fetchone()[0]
    
    print(f"\n✓ Created feat_matrix_v2 with {final_count:,} rows and {n_cols} columns")

    # Quick validation
    print("\nQuick validation:")
    validation = con.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ticker) as n_tickers,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(alpha_composite_v5) as avg_alpha_v5
        FROM feat_matrix_v2
    """).fetchdf()
    print(validation.to_string())

    con.close()
    print(f"\n{'='*70}")
    print("DONE - Feature matrix v2 built successfully")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()