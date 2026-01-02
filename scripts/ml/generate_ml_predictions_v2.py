#!/usr/bin/env python3
"""
generate_ml_predictions_v2.py
=============================
Generate ML v2 predictions using trained models.

This script is designed to run AFTER the main pipeline (run_pipeline.py) completes.
It loads pre-trained XGBoost models and generates predictions for ALL ticker/date
pairs in feat_matrix_v2 (not filtered by ret_5d_f availability).

Key differences from train_xgb_alpha_v2.py:
- No training, CV, or SHAP analysis
- Uses feat_matrix_v2 as source of ticker/date pairs (not feat_targets)
- Generates predictions for recent dates that lack forward returns
- Updates feat_matrix_v2 with alpha_ml_v2_clf column

Usage:
    python scripts/ml/generate_ml_predictions_v2.py --db data/kairos.duckdb
    
Integration:
    Add to run_pipeline.py as Phase 6 after feat_matrix_v2 is built.
"""

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE CONFIGURATION (must match train_xgb_alpha_v2.py exactly)
# =============================================================================

FEATURE_SOURCES = {
    'feat_fundamental': [
        'earnings_yield',
        'fcf_yield',
        'roa',
        'book_to_market',
        'operating_margin',
        'roe',
    ],
    'feat_vol_sizing': [
        'vol_21',
        'vol_63',
        'vol_blend',
    ],
    'feat_beta': [
        'beta_21d',
        'beta_63d',
        'beta_252d',
        'resid_vol_63d',
    ],
    'feat_price_action': [
        'hl_ratio',
        'range_pct',
        'ret_21d',
        'ret_5d',
    ],
    'feat_momentum_v2': [
        'mom_1m',
        'mom_3m',
        'mom_6m',
        'mom_12m',
        'mom_12_1',
        'reversal_1m',
    ],
}

# Flatten to list
FEATURES = []
for cols in FEATURE_SOURCES.values():
    FEATURES.extend(cols)


def load_models(model_dir: Path):
    """Load trained models and feature medians."""
    
    logger.info(f"Loading models from {model_dir}...")
    
    # Load regression model
    reg_path = model_dir / 'model_regression_v2.json'
    if not reg_path.exists():
        raise FileNotFoundError(f"Regression model not found: {reg_path}")
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(reg_path))
    logger.info(f"  Loaded regression model: {reg_path}")
    
    # Load classification model
    clf_path = model_dir / 'model_classification_v2.json'
    if not clf_path.exists():
        raise FileNotFoundError(f"Classification model not found: {clf_path}")
    model_clf = xgb.XGBClassifier()
    model_clf.load_model(str(clf_path))
    logger.info(f"  Loaded classification model: {clf_path}")
    
    # Load feature medians
    medians_path = model_dir / 'feature_medians_v2.json'
    if not medians_path.exists():
        raise FileNotFoundError(f"Feature medians not found: {medians_path}")
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    logger.info(f"  Loaded feature medians: {medians_path}")
    
    return model_reg, model_clf, medians


def load_prediction_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load features for ALL ticker/date pairs in feat_matrix_v2.
    
    Unlike the training script, we don't filter on ret_5d_f IS NOT NULL,
    so we get predictions for recent dates too.
    """
    
    logger.info("Loading ticker/date pairs from feat_matrix_v2...")
    
    # Get all ticker/date pairs from feat_matrix_v2
    base_query = """
        SELECT DISTINCT ticker, date
        FROM feat_matrix_v2
    """
    df = con.execute(base_query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Base grid: {len(df):,} rows from feat_matrix_v2")
    
    # Join each feature table
    for table, cols in FEATURE_SOURCES.items():
        logger.info(f"  Joining {table}...")
        
        cols_str = ', '.join(cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        
        df = df.merge(feat_df, on=['ticker', 'date'], how='left')
        
        # Log coverage
        for col in cols:
            cov = df[col].notna().mean() * 100
            logger.info(f"    {col}: {cov:.1f}%")
    
    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")
    
    return df


def generate_predictions(
    df: pd.DataFrame,
    model_reg: xgb.XGBRegressor,
    model_clf: xgb.XGBClassifier,
    medians: dict
) -> pd.DataFrame:
    """Generate predictions for all rows."""
    
    logger.info("Generating predictions...")
    
    X = df[FEATURES].copy()
    
    # Fill missing with medians from training
    for col in FEATURES:
        median_val = medians.get(col, 0)
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            logger.debug(f"  Filling {n_missing:,} missing values in {col}")
        X[col] = X[col].fillna(median_val)
    
    # Generate predictions
    logger.info("  Running regression model...")
    df['alpha_ml_v2_reg'] = model_reg.predict(X)
    
    logger.info("  Running classification model...")
    df['alpha_ml_v2_clf'] = model_clf.predict_proba(X)[:, 1]
    
    # Summary stats
    logger.info(f"\nPrediction summary:")
    logger.info(f"  alpha_ml_v2_reg: mean={df['alpha_ml_v2_reg'].mean():.4f}, std={df['alpha_ml_v2_reg'].std():.4f}")
    logger.info(f"  alpha_ml_v2_clf: mean={df['alpha_ml_v2_clf'].mean():.4f}, std={df['alpha_ml_v2_clf'].std():.4f}")
    
    return df[['ticker', 'date', 'alpha_ml_v2_reg', 'alpha_ml_v2_clf']]


def save_predictions(con: duckdb.DuckDBPyConnection, predictions_df: pd.DataFrame):
    """Save predictions to feat_alpha_ml_xgb_v2 table."""
    
    logger.info("Saving predictions to feat_alpha_ml_xgb_v2...")
    
    # Drop and recreate table
    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb_v2")
    con.register("predictions_df", predictions_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb_v2 AS 
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)
    
    count = con.execute("SELECT COUNT(*) FROM feat_alpha_ml_xgb_v2").fetchone()[0]
    logger.info(f"  Created feat_alpha_ml_xgb_v2 with {count:,} rows")
    
    # Verify date range
    date_range = con.execute("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM feat_alpha_ml_xgb_v2
    """).fetchone()
    logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")


def update_feat_matrix(con: duckdb.DuckDBPyConnection):
    """Add alpha_ml_v2_clf column to feat_matrix_v2."""
    
    logger.info("Updating feat_matrix_v2 with ML predictions...")
    
    # Get existing indexes on feat_matrix_v2
    indexes = con.execute("""
        SELECT index_name, sql 
        FROM duckdb_indexes() 
        WHERE table_name = 'feat_matrix_v2'
    """).fetchdf()
    
    if len(indexes) > 0:
        logger.info(f"  Found {len(indexes)} indexes to preserve:")
        for _, row in indexes.iterrows():
            logger.info(f"    - {row['index_name']}")
    
    # Drop indexes first (they block table operations)
    for _, row in indexes.iterrows():
        logger.info(f"  Dropping index {row['index_name']}...")
        con.execute(f"DROP INDEX IF EXISTS {row['index_name']}")
    
    # Get columns to exclude ML columns if they exist (we'll re-add them)
    cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'feat_matrix_v2'
    """).fetchdf()['column_name'].tolist()
    
    # Build column list excluding old ML columns
    cols_to_keep = [c for c in cols if c not in ('alpha_ml_v2_clf', 'alpha_ml_v2_reg')]
    cols_str = ', '.join([f'm.{c}' for c in cols_to_keep])
    
    # Create new table with ML columns
    logger.info("  Rebuilding table with ML predictions...")
    con.execute(f"""
        CREATE OR REPLACE TABLE feat_matrix_v2_new AS
        SELECT 
            {cols_str},
            ml.alpha_ml_v2_clf,
            ml.alpha_ml_v2_reg
        FROM feat_matrix_v2 m
        LEFT JOIN feat_alpha_ml_xgb_v2 ml 
            ON m.ticker = ml.ticker AND m.date = ml.date
    """)
    
    # Replace old table
    con.execute("DROP TABLE feat_matrix_v2")
    con.execute("ALTER TABLE feat_matrix_v2_new RENAME TO feat_matrix_v2")
    
    # Recreate indexes
    if len(indexes) > 0:
        logger.info("  Recreating indexes...")
        for _, row in indexes.iterrows():
            logger.info(f"    - {row['index_name']}")
            con.execute(row['sql'])
    
    # Verify
    coverage = con.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN alpha_ml_v2_clf IS NOT NULL THEN 1 ELSE 0 END) as has_ml,
            ROUND(100.0 * SUM(CASE WHEN alpha_ml_v2_clf IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct
        FROM feat_matrix_v2
    """).fetchone()
    
    logger.info(f"  feat_matrix_v2 updated: {coverage[1]:,}/{coverage[0]:,} rows have ML predictions ({coverage[2]}%)")


def validate_predictions(con: duckdb.DuckDBPyConnection):
    """Validate predictions against baseline if available."""
    
    logger.info("\nValidating predictions...")
    
    # Check correlation with baseline v8 (if exists)
    try:
        comparison = con.execute("""
            SELECT 
                CORR(alpha_composite_v8, alpha_ml_v2_clf) as corr_v8_ml,
                COUNT(*) as n
            FROM feat_matrix_v2
            WHERE alpha_composite_v8 IS NOT NULL
              AND alpha_ml_v2_clf IS NOT NULL
        """).fetchone()
        
        if comparison[0] is not None:
            logger.info(f"  Correlation with v8 baseline: {comparison[0]:.4f} (n={comparison[1]:,})")
    except Exception as e:
        logger.warning(f"  Could not compute v8 correlation: {e}")
    
    # Check date coverage vs feat_matrix_v2
    date_check = con.execute("""
        SELECT 
            (SELECT COUNT(DISTINCT date) FROM feat_matrix_v2) as matrix_dates,
            (SELECT COUNT(DISTINCT date) FROM feat_alpha_ml_xgb_v2) as ml_dates
    """).fetchone()
    
    logger.info(f"  Date coverage: {date_check[1]:,} ML dates / {date_check[0]:,} matrix dates")
    
    # Check most recent date has predictions
    recent = con.execute("""
        SELECT 
            m.date,
            COUNT(*) as n_rows,
            SUM(CASE WHEN m.alpha_ml_v2_clf IS NOT NULL THEN 1 ELSE 0 END) as has_ml
        FROM feat_matrix_v2 m
        WHERE m.date = (SELECT MAX(date) FROM feat_matrix_v2)
        GROUP BY m.date
    """).fetchone()
    
    logger.info(f"  Most recent date ({recent[0]}): {recent[2]:,}/{recent[1]:,} rows have ML predictions")


def main():
    parser = argparse.ArgumentParser(description="Generate ML v2 predictions using trained models")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--model-dir", default="scripts/ml/outputs", help="Directory containing trained models")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    logger.info("=" * 70)
    logger.info("ML v2 PREDICTION GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Features: {len(FEATURES)}")
    
    # Load models
    model_reg, model_clf, medians = load_models(model_dir)
    
    # Connect to database
    con = duckdb.connect(args.db)
    
    try:
        # Load data
        df = load_prediction_data(con)
        
        # Generate predictions
        predictions_df = generate_predictions(df, model_reg, model_clf, medians)
        
        # Save to database
        save_predictions(con, predictions_df)
        
        # Update feat_matrix_v2
        update_feat_matrix(con)
        
        # Validate
        validate_predictions(con)
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)
        logger.info("Tables updated:")
        logger.info("  - feat_alpha_ml_xgb_v2 (predictions table)")
        logger.info("  - feat_matrix_v2 (added alpha_ml_v2_clf, alpha_ml_v2_reg)")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()