#!/usr/bin/env python3
"""
backfill_ml_predictions.py
==========================
Backfill ML predictions for recent dates that are missing from feat_targets
(because forward returns aren't available yet).

This is needed for production rebalancing - we need predictions even when
we don't know the outcome yet.

Usage:
    python backfill_ml_predictions.py --db data/kairos.duckdb
"""

import argparse
import duckdb
import pandas as pd
import xgboost as xgb
import json
from pathlib import Path

MODEL_DIR = Path("scripts/ml/outputs")

FEATURES = [
    'earnings_yield', 'fcf_yield', 'roa', 'book_to_market', 'operating_margin', 'roe',
    'vol_21', 'vol_63', 'vol_blend',
    'beta_21d', 'beta_63d', 'beta_252d', 'resid_vol_63d',
    'hl_ratio', 'range_pct', 'ret_21d', 'ret_5d',
    'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1', 'reversal_1m',
]


def main():
    parser = argparse.ArgumentParser(description="Backfill ML predictions for recent dates")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--model-dir", default="scripts/ml/outputs", help="Model directory")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    print("=" * 60)
    print("BACKFILL ML PREDICTIONS FOR RECENT DATES")
    print("=" * 60)
    
    # Load tuned model
    print("\nLoading tuned model...")
    model_clf = xgb.XGBClassifier()
    model_clf.load_model(str(model_dir / 'model_classification_v2_tuned.json'))
    
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(model_dir / 'model_regression_v2_tuned.json'))
    
    with open(model_dir / 'feature_medians_v2_tuned.json') as f:
        medians = json.load(f)
    
    print("  Models loaded successfully")
    
    # Connect to database
    con = duckdb.connect(args.db)
    
    # Find dates missing predictions
    missing_dates = con.execute("""
        SELECT DISTINCT date 
        FROM feat_matrix_v2
        WHERE alpha_ml_v2_tuned_clf IS NULL
          AND date >= '2025-12-01'
        ORDER BY date
    """).fetchdf()['date'].tolist()
    
    if not missing_dates:
        print("\nNo missing dates found. All predictions are current.")
        con.close()
        return
    
    print(f"\nMissing predictions for {len(missing_dates)} dates:")
    for dt in missing_dates:
        print(f"  - {dt}")
    
    # Process each date
    total_updated = 0
    
    for dt in missing_dates:
        print(f"\nProcessing {dt}...")
        
        # Load features for this date from feat_matrix_v2
        features_str = ', '.join([f'"{f}"' for f in FEATURES])
        df = con.execute(f"""
            SELECT ticker, date, {features_str}
            FROM feat_matrix_v2
            WHERE date = '{dt}'
        """).fetchdf()
        
        if len(df) == 0:
            print(f"  No data for {dt}")
            continue
        
        # Prepare features
        X = df[FEATURES].copy()
        for col in FEATURES:
            med = medians.get(col, 0)
            X[col] = X[col].fillna(med)
        
        # Generate predictions
        df['alpha_ml_v2_tuned_clf'] = model_clf.predict_proba(X)[:, 1]
        df['alpha_ml_v2_tuned_reg'] = model_reg.predict(X)
        
        # Update feat_matrix_v2 directly
        print(f"  Updating {len(df)} rows...")
        
        # Batch update using temp table
        con.register('pred_df', df[['ticker', 'date', 'alpha_ml_v2_tuned_clf', 'alpha_ml_v2_tuned_reg']])
        
        con.execute("""
            UPDATE feat_matrix_v2 m
            SET alpha_ml_v2_tuned_clf = p.alpha_ml_v2_tuned_clf,
                alpha_ml_v2_tuned_reg = p.alpha_ml_v2_tuned_reg
            FROM pred_df p
            WHERE m.ticker = p.ticker AND m.date = p.date
        """)
        
        # Also update feat_alpha_ml_xgb_v2_tuned table
        con.execute(f"""
            INSERT INTO feat_alpha_ml_xgb_v2_tuned 
            SELECT ticker, CAST(date AS DATE) as date, 
                   alpha_ml_v2_tuned_reg, alpha_ml_v2_tuned_clf
            FROM pred_df
            ON CONFLICT DO NOTHING
        """)
        
        total_updated += len(df)
        print(f"  Done: {len(df)} predictions generated")
    
    # Verify results
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    result = con.execute("""
        SELECT date, 
               COUNT(*) as total, 
               COUNT(alpha_ml_v2_tuned_clf) as has_pred,
               ROUND(100.0 * COUNT(alpha_ml_v2_tuned_clf) / COUNT(*), 1) as pct
        FROM feat_matrix_v2
        WHERE date >= '2025-12-20'
        GROUP BY date 
        ORDER BY date DESC
        LIMIT 15
    """).fetchdf()
    print(result.to_string(index=False))
    
    con.close()
    
    print(f"\n✓ Total rows updated: {total_updated:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()