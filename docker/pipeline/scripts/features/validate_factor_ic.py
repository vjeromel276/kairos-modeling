#!/usr/bin/env python3
"""
validate_factor_ic.py
=====================
Validate Information Coefficients for all v2 factors.

Tests each factor against forward returns to confirm predictive power
matches Phase 4 investigation findings.

Expected ICs:
- Value factors: 0.014-0.017
- Momentum (12-1): 0.012
- Reversal (1m): 0.008
- Insider: 0.005
- Quality: 0.005-0.010
- Composite v5: 0.015-0.025
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from scipy import stats

def compute_ic(factor: pd.Series, forward_ret: pd.Series) -> tuple:
    """Compute Information Coefficient (Spearman rank correlation)."""
    mask = factor.notna() & forward_ret.notna()
    if mask.sum() < 100:
        return np.nan, np.nan, 0
    
    corr, pval = stats.spearmanr(factor[mask], forward_ret[mask])
    return corr, pval, mask.sum()

def main():
    parser = argparse.ArgumentParser(description="Validate factor ICs")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--start-date", default="2010-01-01", help="Start date for validation")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for validation")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("FACTOR IC VALIDATION")
    print(f"{'='*70}\n")
    print(f"Period: {args.start_date} to {args.end_date}")

    con = duckdb.connect(args.db)

    # Check available tables
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()['table_name'].tolist()
    
    print(f"\nAvailable factor tables:")
    factor_tables = [t for t in tables if t.startswith('feat_')]
    for t in factor_tables:
        print(f"  - {t}")

    # Get forward returns from feat_targets or compute from SEP
    print("\nLoading forward returns...")
    
    if 'feat_targets' in tables:
        returns = con.execute(f"""
            SELECT ticker, date, ret_5d_f as fwd_ret
            FROM feat_targets
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
    else:
        print("  Computing from SEP...")
        returns = con.execute(f"""
            WITH price_data AS (
                SELECT 
                    ticker,
                    date,
                    closeadj,
                    LEAD(closeadj, 5) OVER (PARTITION BY ticker ORDER BY date) AS price_5d_fwd
                FROM sep
                WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
            )
            SELECT 
                ticker,
                date,
                (price_5d_fwd / closeadj) - 1 AS fwd_ret
            FROM price_data
            WHERE price_5d_fwd IS NOT NULL
        """).fetchdf()
    
    print(f"  Loaded {len(returns):,} return observations")

    # Results storage
    results = []

    # Test each factor table
    print(f"\n{'='*70}")
    print("FACTOR IC RESULTS")
    print(f"{'='*70}")
    print(f"{'Factor':<35} {'IC':>10} {'t-stat':>10} {'N Obs':>12}")
    print("-" * 70)

    # Value factors
    if 'feat_value_v2' in tables:
        factors = con.execute(f"""
            SELECT ticker, date, 
                   earnings_yield, book_yield, ebitda_yield, sales_yield,
                   value_composite_z, value_quality_adj
            FROM feat_value_v2
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
        
        merged = factors.merge(returns, on=['ticker', 'date'])
        
        for col in ['earnings_yield', 'book_yield', 'ebitda_yield', 'sales_yield', 
                    'value_composite_z', 'value_quality_adj']:
            ic, pval, n = compute_ic(merged[col], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{col:<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            results.append({'factor': col, 'ic': ic, 'tstat': tstat, 'n': n})

    # Momentum factors
    if 'feat_momentum_v2' in tables:
        factors = con.execute(f"""
            SELECT ticker, date, 
                   mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1,
                   reversal_1m, momentum_composite_z
            FROM feat_momentum_v2
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
        
        merged = factors.merge(returns, on=['ticker', 'date'])
        
        for col in ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1',
                    'reversal_1m', 'momentum_composite_z']:
            ic, pval, n = compute_ic(merged[col], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{col:<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            results.append({'factor': col, 'ic': ic, 'tstat': tstat, 'n': n})

    # Quality factors
    if 'feat_quality_v2' in tables:
        factors = con.execute(f"""
            SELECT ticker, date, 
                   roe, roa, accruals, quality_composite_z
            FROM feat_quality_v2
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
        
        merged = factors.merge(returns, on=['ticker', 'date'])
        
        for col in ['roe', 'roa', 'accruals', 'quality_composite_z']:
            ic, pval, n = compute_ic(merged[col], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{col:<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            results.append({'factor': col, 'ic': ic, 'tstat': tstat, 'n': n})

    # Insider factors
    if 'feat_insider' in tables:
        factors = con.execute(f"""
            SELECT ticker, date, 
                   net_buy_signal, insider_composite_z
            FROM feat_insider
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
        
        merged = factors.merge(returns, on=['ticker', 'date'])
        
        for col in ['net_buy_signal', 'insider_composite_z']:
            ic, pval, n = compute_ic(merged[col], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{col:<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            results.append({'factor': col, 'ic': ic, 'tstat': tstat, 'n': n})

    # Composite v5
    if 'feat_composite_v5' in tables:
        factors = con.execute(f"""
            SELECT ticker, date, 
                   alpha_composite_v5, alpha_composite_v5_regime
            FROM feat_composite_v5
            WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
        """).fetchdf()
        
        merged = factors.merge(returns, on=['ticker', 'date'])
        
        for col in ['alpha_composite_v5', 'alpha_composite_v5_regime']:
            ic, pval, n = compute_ic(merged[col], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{col:<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            results.append({'factor': col, 'ic': ic, 'tstat': tstat, 'n': n})

    # Compare to old alpha
    if 'feat_composite_v33_regime' in tables or 'feat_matrix' in tables:
        print("\n" + "-" * 70)
        print("COMPARISON TO EXISTING ALPHA")
        print("-" * 70)
        
        try:
            if 'feat_composite_v33_regime' in tables:
                old_alpha = con.execute(f"""
                    SELECT ticker, date, alpha_composite_v33_regime
                    FROM feat_composite_v33_regime
                    WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
                """).fetchdf()
            else:
                old_alpha = con.execute(f"""
                    SELECT ticker, date, alpha_composite_v33_regime
                    FROM feat_matrix
                    WHERE date BETWEEN '{args.start_date}' AND '{args.end_date}'
                """).fetchdf()
            
            merged = old_alpha.merge(returns, on=['ticker', 'date'])
            ic, pval, n = compute_ic(merged['alpha_composite_v33_regime'], merged['fwd_ret'])
            tstat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if not np.isnan(ic) else np.nan
            print(f"{'alpha_composite_v33_regime (OLD)':<35} {ic:>10.4f} {tstat:>10.2f} {n:>12,}")
            
        except Exception as e:
            print(f"  Could not load old alpha: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        best = results_df.loc[results_df['ic'].idxmax()]
        print(f"\nBest factor: {best['factor']} with IC = {best['ic']:.4f}")
        
        # Calculate expected composite IC (assuming independence)
        composite_factors = ['value_composite_z', 'momentum_composite_z', 
                           'quality_composite_z', 'insider_composite_z']
        composite_ics = results_df[results_df['factor'].isin(composite_factors)]['ic'].dropna()
        if len(composite_ics) > 0:
            # Rough estimate: IC of composite ≈ sqrt(sum(IC^2)) for uncorrelated factors
            expected_ic = np.sqrt((composite_ics**2).sum())
            print(f"Expected composite IC (if uncorrelated): {expected_ic:.4f}")

    con.close()
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
