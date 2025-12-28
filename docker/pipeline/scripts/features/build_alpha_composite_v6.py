#!/usr/bin/env python3
"""
build_alpha_composite_v6.py
===========================
Build improved alpha composite with weights based on ACTUAL IC validation results.

CORRECTED WEIGHTS based on IC validation (not investigation estimates):

Factor              | Actual IC | Old Weight | New Weight
--------------------|-----------|------------|------------
Quality composite   | 0.0223    | 0.15       | 0.40
Reversal (1m)       | 0.0198    | 0.16       | 0.30
Momentum composite  | 0.0132    | 0.24       | 0.20
Value composite     | 0.0036    | 0.35       | 0.10
Insider             | -0.0167   | 0.10       | 0.00 (REMOVED)

Expected composite IC: 0.018-0.020 (vs 0.0126 for v5)

Key changes from v5:
1. Quality is now the dominant factor (was severely underweighted)
2. Insider signal REMOVED (negative IC was hurting performance)
3. Value demoted (much weaker than expected)
4. Reversal promoted (very strong signal)

Output table: feat_composite_v6
Columns: ticker, date, alpha_composite_v6, alpha_composite_v6_regime
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
    parser = argparse.ArgumentParser(description="Build alpha composite v6 with corrected weights")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    # Weights based on actual IC validation results
    parser.add_argument("--quality-weight", type=float, default=0.40, help="Weight for quality factor (IC=0.0223)")
    parser.add_argument("--reversal-weight", type=float, default=0.30, help="Weight for reversal factor (IC=0.0198)")
    parser.add_argument("--momentum-weight", type=float, default=0.20, help="Weight for momentum factor (IC=0.0132)")
    parser.add_argument("--value-weight", type=float, default=0.10, help="Weight for value factor (IC=0.0036)")
    parser.add_argument("--insider-weight", type=float, default=0.00, help="Weight for insider factor (IC=-0.0167, REMOVED)")
    args = parser.parse_args()

    # Normalize weights
    total_weight = args.quality_weight + args.reversal_weight + args.momentum_weight + args.value_weight + args.insider_weight
    if total_weight == 0:
        total_weight = 1.0
    
    w_qual = args.quality_weight / total_weight
    w_rev = args.reversal_weight / total_weight
    w_mom = args.momentum_weight / total_weight
    w_val = args.value_weight / total_weight
    w_ins = args.insider_weight / total_weight

    print(f"\n{'='*60}")
    print("BUILD ALPHA COMPOSITE V6 (IC-Corrected Weights)")
    print(f"{'='*60}\n")
    print("Factor weights based on ACTUAL IC validation:")
    print(f"  Quality:   {w_qual:.2f} (IC=0.0223, was 0.15)")
    print(f"  Reversal:  {w_rev:.2f} (IC=0.0198, was 0.16)")
    print(f"  Momentum:  {w_mom:.2f} (IC=0.0132, was 0.24)")
    print(f"  Value:     {w_val:.2f} (IC=0.0036, was 0.35)")
    print(f"  Insider:   {w_ins:.2f} (IC=-0.017, REMOVED)")
    
    if w_ins > 0:
        print("\n  ⚠ WARNING: Insider weight > 0 but IC is negative!")
        print("    Consider setting --insider-weight 0")

    con = duckdb.connect(args.db)

    # Check that all required tables exist
    print("\nChecking required factor tables...")
    required_tables = ['feat_value_v2', 'feat_momentum_v2', 'feat_quality_v2']
    if w_ins > 0:
        required_tables.append('feat_insider')
    
    existing = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()['table_name'].tolist()
    
    missing = [t for t in required_tables if t not in existing]
    if missing:
        print(f"  ⚠ Missing tables: {missing}")
        raise ValueError(f"Missing required tables: {missing}")
    
    print(f"  ✓ All required tables present")

    # Build composite from available factors
    print("\nBuilding composite alpha...")
    
    # Start with trading dates from SEP as base
    print("  Getting trading dates...")
    base = con.execute("""
        SELECT DISTINCT ticker, date 
        FROM sep_base 
        WHERE date >= '2000-01-01'
    """).fetchdf()
    print(f"  Base: {len(base):,} ticker-date pairs")

    # Join quality factors (PRIMARY - highest IC)
    print("  Joining quality factors (IC=0.0223)...")
    quality = con.execute("""
        SELECT ticker, date, quality_composite_z
        FROM feat_quality_v2
    """).fetchdf()
    base = base.merge(quality, on=['ticker', 'date'], how='left')
    print(f"    Quality coverage: {base['quality_composite_z'].notna().mean()*100:.1f}%")

    # Join momentum factors (for reversal_1m and momentum_composite)
    print("  Joining momentum factors...")
    momentum = con.execute("""
        SELECT ticker, date, 
               mom_12_1 as momentum_raw,
               reversal_1m as reversal_raw,
               momentum_composite_z
        FROM feat_momentum_v2
    """).fetchdf()
    base = base.merge(momentum, on=['ticker', 'date'], how='left')
    print(f"    Momentum coverage: {base['momentum_raw'].notna().mean()*100:.1f}%")
    print(f"    Reversal coverage: {base['reversal_raw'].notna().mean()*100:.1f}%")

    # Join value factors
    print("  Joining value factors (IC=0.0036)...")
    value = con.execute("""
        SELECT ticker, date, value_composite_z
        FROM feat_value_v2
    """).fetchdf()
    base = base.merge(value, on=['ticker', 'date'], how='left')
    print(f"    Value coverage: {base['value_composite_z'].notna().mean()*100:.1f}%")

    # Optionally join insider factors (but weight is 0 by default)
    if w_ins > 0 and 'feat_insider' in existing:
        print("  Joining insider factors (IC=-0.017, NOT RECOMMENDED)...")
        insider = con.execute("""
            SELECT ticker, date, insider_composite_z
            FROM feat_insider
        """).fetchdf()
        base = base.merge(insider, on=['ticker', 'date'], how='left')
        print(f"    Insider coverage: {base['insider_composite_z'].notna().mean()*100:.1f}%")
    else:
        base['insider_composite_z'] = 0.0

    # Cross-sectional z-score reversal (already have momentum_composite_z and quality_composite_z)
    print("\nComputing cross-sectional z-scores for reversal...")
    
    def zscore_reversal(group):
        group['reversal_z'] = zscore_winsorize(group['reversal_raw'])
        return group
    
    base = base.groupby('date', group_keys=False).apply(zscore_reversal)

    # Compute composite with corrected weights
    print("Computing IC-weighted composite...")
    
    # Fill NaN with 0 (neutral) for missing factors
    for col in ['quality_composite_z', 'reversal_z', 'momentum_composite_z', 'value_composite_z', 'insider_composite_z']:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    # Weighted sum with IC-corrected weights
    base['alpha_composite_v6'] = (
        w_qual * base['quality_composite_z'] +
        w_rev * base['reversal_z'] +
        w_mom * base['momentum_composite_z'] +
        w_val * base['value_composite_z'] +
        w_ins * base['insider_composite_z']
    )

    # Final z-score of composite
    print("Final z-scoring of composite...")
    base = base.groupby('date', group_keys=False).apply(
        lambda g: g.assign(alpha_composite_v6=zscore_winsorize(g['alpha_composite_v6']))
    )

    # Add regime conditioning
    print("\nChecking for regime data...")
    try:
        # First check what columns exist in feat_vol_sizing
        vol_cols = con.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'feat_vol_sizing'
        """).fetchdf()['column_name'].tolist()
        print(f"  feat_vol_sizing columns: {vol_cols}")
        
        # Check if we have the columns we need
        has_vol_blend = 'vol_blend' in vol_cols
        has_vol_21 = 'vol_21' in vol_cols
        
        if has_vol_blend or has_vol_21:
            vol_col = 'vol_blend' if has_vol_blend else 'vol_21'
            
            # Get vol data 
            vol_data = con.execute(f"""
                SELECT DISTINCT ticker, date, {vol_col} as vol_measure
                FROM feat_vol_sizing
                WHERE {vol_col} IS NOT NULL
            """).fetchdf()
            
            # Get 21-day returns from feat_price_action or compute from momentum
            # Check which table has ret_21d
            ret_source = None
            for tbl in ['feat_price_action', 'feat_stat', 'feat_momentum_v2']:
                try:
                    cols = con.execute(f"""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = '{tbl}'
                    """).fetchdf()['column_name'].tolist()
                    if 'ret_21d' in cols:
                        ret_source = tbl
                        break
                    elif tbl == 'feat_momentum_v2' and 'mom_1m' in cols:
                        ret_source = tbl  # Can use mom_1m as proxy
                        break
                except:
                    continue
            
            if ret_source:
                print(f"  Getting returns from {ret_source}...")
                if ret_source == 'feat_momentum_v2':
                    # Use mom_1m as proxy for recent returns
                    mkt_ret = con.execute(f"""
                        SELECT date, AVG(mom_1m) as mkt_ret
                        FROM {ret_source}
                        GROUP BY date
                    """).fetchdf()
                else:
                    mkt_ret = con.execute(f"""
                        SELECT date, AVG(ret_21d) as mkt_ret
                        FROM {ret_source}
                        GROUP BY date
                    """).fetchdf()
                
                # Aggregate vol to daily level
                vol_daily = vol_data.groupby('date')['vol_measure'].mean().reset_index()
                vol_daily = vol_daily.merge(mkt_ret, on='date', how='left')
            else:
                print("  No return data found - using vol-only regime")
                vol_daily = vol_data.groupby('date')['vol_measure'].mean().reset_index()
                vol_daily['mkt_ret'] = 0  # Neutral
            
            # Compute regime based on vol and market return
            def classify_regime(row):
                vol = row['vol_measure']
                ret = row.get('mkt_ret', 0) 
                if pd.isna(ret):
                    ret = 0
                
                if pd.isna(vol):
                    return 'unknown'
                elif vol > 0.30:  # High vol threshold
                    return 'high_vol_bear' if ret < -0.02 else 'high_vol_bull'
                elif vol < 0.15:  # Low vol threshold
                    return 'low_vol'
                else:  # Normal vol
                    if ret > 0.02:
                        return 'normal_vol_bull'
                    elif ret < -0.02:
                        return 'normal_vol_bear'
                    else:
                        return 'normal_vol_neutral'
            
            vol_daily['regime'] = vol_daily.apply(classify_regime, axis=1)
            
            base = base.merge(vol_daily[['date', 'regime']], on='date', how='left')
            
            # Regime multipliers (from Phase 4 analysis)
            regime_mult = {
                'high_vol_bear': 1.2,
                'high_vol_bull': 1.1,
                'normal_vol_bull': 1.1,
                'normal_vol_neutral': 0.5,
                'normal_vol_bear': 0.8,
                'low_vol': 0.8,
                'unknown': 1.0
            }
            base['regime_mult'] = base['regime'].map(regime_mult).fillna(1.0)
            base['alpha_composite_v6_regime'] = base['alpha_composite_v6'] * base['regime_mult']
            print(f"  ✓ Applied regime multipliers based on {vol_col}")
            
            # Show regime distribution
            regime_dist = base['regime'].value_counts()
            print(f"  Regime distribution:")
            for regime, count in regime_dist.head(10).items():
                pct = count / len(base) * 100
                print(f"    {regime}: {count:,} ({pct:.1f}%)")
        else:
            print("  No vol data available - using base alpha")
            base['alpha_composite_v6_regime'] = base['alpha_composite_v6']
            
    except Exception as e:
        print(f"  ⚠ Could not compute regime: {e}")
        base['alpha_composite_v6_regime'] = base['alpha_composite_v6']

    # Select final columns
    df = base[['ticker', 'date', 'alpha_composite_v6', 'alpha_composite_v6_regime']].copy()
    
    # Drop rows with no alpha
    df = df.dropna(subset=['alpha_composite_v6'])
    print(f"\nFinal rows: {len(df):,}")

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_composite_v6")
    con.execute("""
        CREATE TABLE feat_composite_v6 (
            ticker VARCHAR,
            date DATE,
            alpha_composite_v6 DOUBLE,
            alpha_composite_v6_regime DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_composite_v6 SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_composite_v6_ticker_date ON feat_composite_v6(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_composite_v6_date ON feat_composite_v6(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_composite_v6").fetchone()[0]
    print(f"\n✓ Created feat_composite_v6 with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(alpha_composite_v6) as avg_alpha,
            STDDEV(alpha_composite_v6) as std_alpha,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY alpha_composite_v6) as p5,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY alpha_composite_v6) as p95
        FROM feat_composite_v6
    """).fetchdf()
    print(stats.to_string())

    # Expected IC calculation
    print("\n" + "="*60)
    print("EXPECTED IC IMPROVEMENT")
    print("="*60)
    print(f"\nv5 weights -> IC = 0.0126")
    print(f"v6 weights (IC-corrected):")
    expected_ic = (
        w_qual * 0.0223 +
        w_rev * 0.0198 +
        w_mom * 0.0132 +
        w_val * 0.0036 +
        w_ins * (-0.0167)
    )
    print(f"  Expected IC ≈ {expected_ic:.4f}")
    print(f"  Expected improvement: {(expected_ic/0.0126 - 1)*100:.1f}%")

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Alpha composite v6 built successfully")
    print(f"{'='*60}\n")
    print("Next: Run backtest to validate:")
    print(f"  python scripts/backtesting/backtest_academic_strategy_risk4.py \\")
    print(f"    --db data/kairos.duckdb \\")
    print(f"    --alpha-column alpha_composite_v6 \\")
    print(f"    --target-column ret_5d_f \\")
    print(f"    --top-n 75 --rebalance-every 5 --target-vol 0.20")
    print()


if __name__ == "__main__":
    main()