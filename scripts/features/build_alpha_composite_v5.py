#!/usr/bin/env python3
"""
build_alpha_composite_v5.py
===========================
Build new alpha composite combining all v2 factors with real predictive power.

Based on Phase 4 Investigation findings:
Factor              | IC      | Weight (IC-proportional)
--------------------|---------|-------------------------
EBITDA yield        | 0.0173  | 0.35
Book yield          | 0.0142  | -
Earnings yield      | 0.0141  | -
Value composite     | ~0.015  | (included above)
12-1 month momentum | 0.0118  | 0.24
1-month reversal    | 0.0077  | 0.16
Insider net buying  | 0.0049  | 0.10
Quality composite   | ~0.007  | 0.15 (estimated)
--------------------|---------|-------------------------
Total               | ~0.05   | 1.00

Expected composite IC: 0.015-0.025 (75-125x improvement over current 0.0002)

Output table: feat_composite_v5
Columns: ticker, date, alpha_composite_v5, alpha_composite_v5_regime
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
    parser = argparse.ArgumentParser(description="Build alpha composite v5")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--value-weight", type=float, default=0.35, help="Weight for value factor")
    parser.add_argument("--momentum-weight", type=float, default=0.24, help="Weight for momentum factor")
    parser.add_argument("--reversal-weight", type=float, default=0.16, help="Weight for reversal factor")
    parser.add_argument("--quality-weight", type=float, default=0.15, help="Weight for quality factor")
    parser.add_argument("--insider-weight", type=float, default=0.10, help="Weight for insider factor")
    args = parser.parse_args()

    # Normalize weights
    total_weight = args.value_weight + args.momentum_weight + args.reversal_weight + args.quality_weight + args.insider_weight
    w_val = args.value_weight / total_weight
    w_mom = args.momentum_weight / total_weight
    w_rev = args.reversal_weight / total_weight
    w_qual = args.quality_weight / total_weight
    w_ins = args.insider_weight / total_weight

    print(f"\n{'='*60}")
    print("BUILD ALPHA COMPOSITE V5")
    print(f"{'='*60}\n")
    print("Factor weights (normalized):")
    print(f"  Value:     {w_val:.2f}")
    print(f"  Momentum:  {w_mom:.2f}")
    print(f"  Reversal:  {w_rev:.2f}")
    print(f"  Quality:   {w_qual:.2f}")
    print(f"  Insider:   {w_ins:.2f}")

    con = duckdb.connect(args.db)

    # Check that all required tables exist
    print("\nChecking required factor tables...")
    required_tables = ['feat_value_v2', 'feat_momentum_v2', 'feat_quality_v2', 'feat_insider']
    
    existing = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()['table_name'].tolist()
    
    missing = [t for t in required_tables if t not in existing]
    if missing:
        print(f"  ⚠ Missing tables: {missing}")
        print("  Run the following scripts first:")
        for t in missing:
            print(f"    python scripts/features/build_{t.replace('feat_', '')}.py --db {args.db}")
        print("\n  Continuing with available factors...")
    
    available = [t for t in required_tables if t in existing]
    print(f"  ✓ Available tables: {available}")

    # Build composite from available factors
    print("\nBuilding composite alpha...")
    
    # Start with trading dates from SEP as base
    print("  Getting trading dates...")
    base = con.execute("""
        SELECT DISTINCT ticker, date 
        FROM sep 
        WHERE date >= '2000-01-01'
    """).fetchdf()
    print(f"  Base: {len(base):,} ticker-date pairs")

    # Join each factor table
    if 'feat_value_v2' in available:
        print("  Joining value factors...")
        value = con.execute("""
            SELECT ticker, date, value_quality_adj as value_z
            FROM feat_value_v2
        """).fetchdf()
        base = base.merge(value, on=['ticker', 'date'], how='left')
        print(f"    Value coverage: {base['value_z'].notna().mean()*100:.1f}%")
    else:
        base['value_z'] = np.nan

    if 'feat_momentum_v2' in available:
        print("  Joining momentum factors...")
        momentum = con.execute("""
            SELECT ticker, date, 
                   mom_12_1 as momentum_raw,
                   reversal_1m as reversal_raw
            FROM feat_momentum_v2
        """).fetchdf()
        base = base.merge(momentum, on=['ticker', 'date'], how='left')
        print(f"    Momentum coverage: {base['momentum_raw'].notna().mean()*100:.1f}%")
    else:
        base['momentum_raw'] = np.nan
        base['reversal_raw'] = np.nan

    if 'feat_quality_v2' in available:
        print("  Joining quality factors...")
        quality = con.execute("""
            SELECT ticker, date, quality_composite_z as quality_z
            FROM feat_quality_v2
        """).fetchdf()
        base = base.merge(quality, on=['ticker', 'date'], how='left')
        print(f"    Quality coverage: {base['quality_z'].notna().mean()*100:.1f}%")
    else:
        base['quality_z'] = np.nan

    if 'feat_insider' in available:
        print("  Joining insider factors...")
        insider = con.execute("""
            SELECT ticker, date, insider_composite_z as insider_z
            FROM feat_insider
        """).fetchdf()
        base = base.merge(insider, on=['ticker', 'date'], how='left')
        print(f"    Insider coverage: {base['insider_z'].notna().mean()*100:.1f}%")
    else:
        base['insider_z'] = np.nan

    # Cross-sectional z-score momentum and reversal
    print("\nComputing cross-sectional z-scores...")
    
    def zscore_group(group):
        if 'momentum_raw' in group.columns:
            group['momentum_z'] = zscore_winsorize(group['momentum_raw'])
        if 'reversal_raw' in group.columns:
            group['reversal_z'] = zscore_winsorize(group['reversal_raw'])
        return group
    
    base = base.groupby('date', group_keys=False).apply(zscore_group)

    # Compute composite
    print("Computing weighted composite...")
    
    # Fill NaN with 0 (neutral) for missing factors
    for col in ['value_z', 'momentum_z', 'reversal_z', 'quality_z', 'insider_z']:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    base['alpha_composite_v5'] = (
        w_val * base['value_z'] +
        w_mom * base['momentum_z'] +
        w_rev * base['reversal_z'] +
        w_qual * base['quality_z'] +
        w_ins * base['insider_z']
    )

    # Final z-score of composite
    print("Final z-scoring of composite...")
    base = base.groupby('date', group_keys=False).apply(
        lambda g: g.assign(alpha_composite_v5=zscore_winsorize(g['alpha_composite_v5']))
    )

    # Add regime conditioning (optional - check if regime data exists)
    print("\nChecking for regime data...")
    if 'feat_vol_sizing' in existing or 'spy_regime' in existing:
        print("  Regime data found - adding regime-conditional alpha")
        # Load regime from vol_sizing or spy_regime table
        try:
            regime = con.execute("""
                SELECT date, regime 
                FROM feat_vol_sizing
            """).fetchdf()
            base = base.merge(regime, on='date', how='left')
            
            # Regime multipliers (from Phase 4 analysis)
            # high_vol_bear: LO Sharpe 2.46
            # high_vol_bull: LO Sharpe 1.82
            # normal_vol_bull: LO Sharpe 2.19
            # normal_vol_neutral: LO Sharpe 0.20
            regime_mult = {
                'high_vol_bear': 1.2,
                'high_vol_bull': 1.1,
                'normal_vol_bull': 1.1,
                'normal_vol_neutral': 0.5,  # Reduce exposure
                'low_vol': 0.8
            }
            base['regime_mult'] = base['regime'].map(regime_mult).fillna(1.0)
            base['alpha_composite_v5_regime'] = base['alpha_composite_v5'] * base['regime_mult']
        except Exception as e:
            print(f"  ⚠ Could not load regime: {e}")
            base['alpha_composite_v5_regime'] = base['alpha_composite_v5']
    else:
        print("  No regime data - using base alpha")
        base['alpha_composite_v5_regime'] = base['alpha_composite_v5']

    # Select final columns
    df = base[['ticker', 'date', 'alpha_composite_v5', 'alpha_composite_v5_regime']].copy()
    
    # Drop rows with no alpha
    df = df.dropna(subset=['alpha_composite_v5'])
    print(f"\nFinal rows: {len(df):,}")

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_composite_v5")
    con.execute("""
        CREATE TABLE feat_composite_v5 (
            ticker VARCHAR,
            date DATE,
            alpha_composite_v5 DOUBLE,
            alpha_composite_v5_regime DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_composite_v5 SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_composite_v5_ticker_date ON feat_composite_v5(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_composite_v5_date ON feat_composite_v5(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_composite_v5").fetchone()[0]
    print(f"\n✓ Created feat_composite_v5 with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(alpha_composite_v5) as avg_alpha,
            STDDEV(alpha_composite_v5) as std_alpha,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY alpha_composite_v5) as p5,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY alpha_composite_v5) as p95
        FROM feat_composite_v5
    """).fetchdf()
    print(stats.to_string())

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Alpha composite v5 built successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
