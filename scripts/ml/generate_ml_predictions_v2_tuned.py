#!/usr/bin/env python3
"""
generate_ml_predictions_v2_tuned.py
===================================
Generate ML v2 TUNED predictions using Optuna-optimized models.

This script is designed to run AFTER generate_ml_predictions_v2.py and BEFORE
the matrix builder runs.

It loads pre-trained XGBoost models (tuned via Optuna) and generates predictions
for ALL ticker/date pairs in feat_targets.

Creates: feat_alpha_ml_xgb_v2_tuned table
    - alpha_ml_v2_tuned_reg (regression predictions)
    - alpha_ml_v2_tuned_clf (classification probabilities)

The matrix builder (build_feature_matrix_v2.py) handles joining this table
into feat_matrix_v2.

Usage:
    python scripts/ml/generate_ml_predictions_v2_tuned.py --db data/kairos.duckdb
    
Integration:
    Add to run_pipeline.py Phase 6, runs after generate_ml_predictions_v2.py
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
# FEATURE CONFIGURATION (must match train_xgb_alpha_v2_tuned.py exactly)
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
    """Load trained TUNED models and feature medians."""
    
    logger.info(f"Loading TUNED models from {model_dir}...")
    
    # Load regression model (tuned)
    reg_path = model_dir / 'model_regression_v2_tuned.json'
    if not reg_path.exists():
        raise FileNotFoundError(f"Tuned regression model not found: {reg_path}")
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(reg_path))
    logger.info(f"  Loaded tuned regression model: {reg_path}")
    
    # Load classification model (tuned)
    clf_path = model_dir / 'model_classification_v2_tuned.json'
    if not clf_path.exists():
        raise FileNotFoundError(f"Tuned classification model not found: {clf_path}")
    model_clf = xgb.XGBClassifier()
    model_clf.load_model(str(clf_path))
    logger.info(f"  Loaded tuned classification model: {clf_path}")
    
    # Load feature medians (tuned version if exists, else fall back to base)
    medians_path = model_dir / 'feature_medians_v2_tuned.json'
    if not medians_path.exists():
        # Fall back to base medians (features are the same)
        medians_path = model_dir / 'feature_medians_v2.json'
        if not medians_path.exists():
            raise FileNotFoundError(f"Feature medians not found in {model_dir}")
        logger.info(f"  Using base feature medians: {medians_path}")
    else:
        logger.info(f"  Loaded tuned feature medians: {medians_path}")
    
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    
    return model_reg, model_clf, medians


def load_prediction_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load features for ALL ticker/date pairs that have feature data.
    
    Uses feat_targets as the base (has all ticker/date pairs we care about),
    then joins feature tables.
    """
    
    logger.info("Loading ticker/date pairs from feat_targets...")
    
    # Get all ticker/date pairs from feat_targets
    base_query = """
        SELECT DISTINCT ticker, date
        FROM sep_base_academic
        WHERE date >= '2015-01-01'
    """
    df = con.execute(base_query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Base grid: {len(df):,} rows from feat_targets")
    
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
    """Generate predictions for all rows using TUNED models."""
    
    logger.info("Generating TUNED predictions...")
    
    X = df[FEATURES].copy()
    
    # Fill missing with medians from training
    for col in FEATURES:
        median_val = medians.get(col, 0)
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            logger.debug(f"  Filling {n_missing:,} missing values in {col}")
        X[col] = X[col].fillna(median_val)
    
    # Generate predictions
    logger.info("  Running tuned regression model...")
    df['alpha_ml_v2_tuned_reg'] = model_reg.predict(X)
    
    logger.info("  Running tuned classification model...")
    df['alpha_ml_v2_tuned_clf'] = model_clf.predict_proba(X)[:, 1]
    
    # Summary stats
    logger.info(f"\nPrediction summary:")
    logger.info(f"  alpha_ml_v2_tuned_reg: mean={df['alpha_ml_v2_tuned_reg'].mean():.4f}, std={df['alpha_ml_v2_tuned_reg'].std():.4f}")
    logger.info(f"  alpha_ml_v2_tuned_clf: mean={df['alpha_ml_v2_tuned_clf'].mean():.4f}, std={df['alpha_ml_v2_tuned_clf'].std():.4f}")
    
    return df[['ticker', 'date', 'alpha_ml_v2_tuned_reg', 'alpha_ml_v2_tuned_clf']]


def save_predictions(con: duckdb.DuckDBPyConnection, predictions_df: pd.DataFrame):
    """Save predictions to feat_alpha_ml_xgb_v2_tuned table."""
    
    logger.info("Saving predictions to feat_alpha_ml_xgb_v2_tuned...")
    
    # Drop and recreate table
    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb_v2_tuned")
    con.register("predictions_df", predictions_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb_v2_tuned AS 
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)
    
    count = con.execute("SELECT COUNT(*) FROM feat_alpha_ml_xgb_v2_tuned").fetchone()[0]
    logger.info(f"  Created feat_alpha_ml_xgb_v2_tuned with {count:,} rows")
    
    # Create index for join performance
    con.execute("CREATE INDEX IF NOT EXISTS idx_ml_v2_tuned_ticker_date ON feat_alpha_ml_xgb_v2_tuned(ticker, date)")
    
    # Verify date range
    date_range = con.execute("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM feat_alpha_ml_xgb_v2_tuned
    """).fetchone()
    logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")


def validate_predictions(con: duckdb.DuckDBPyConnection):
    """Validate tuned predictions and compare to base v2 if available."""
    
    logger.info("\nValidating tuned predictions...")
    
    # Check IC against returns
    try:
        correlation = con.execute("""
            SELECT 
                CORR(ml.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_tuned_clf,
                CORR(ml.alpha_ml_v2_tuned_reg, t.ret_5d_f) as ic_tuned_reg,
                COUNT(*) as n
            FROM feat_alpha_ml_xgb_v2_tuned ml
            JOIN feat_targets t ON ml.ticker = t.ticker AND ml.date = t.date
            WHERE t.ret_5d_f IS NOT NULL
        """).fetchone()
        
        if correlation[0] is not None:
            logger.info(f"  IC (tuned clf vs ret_5d_f): {correlation[0]:.4f}")
            logger.info(f"  IC (tuned reg vs ret_5d_f): {correlation[1]:.4f}")
            logger.info(f"  (n = {correlation[2]:,})")
    except Exception as e:
        logger.warning(f"  Could not compute IC: {e}")
    
    # Compare to base v2 if available
    try:
        base_exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'feat_alpha_ml_xgb_v2'
        """).fetchone()[0] > 0
        
        if base_exists:
            comparison = con.execute("""
                SELECT 
                    CORR(base.alpha_ml_v2_clf, t.ret_5d_f) as ic_base,
                    CORR(tuned.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_tuned,
                    CORR(base.alpha_ml_v2_clf, tuned.alpha_ml_v2_tuned_clf) as correlation,
                    COUNT(*) as n
                FROM feat_alpha_ml_xgb_v2 base
                JOIN feat_alpha_ml_xgb_v2_tuned tuned 
                    ON base.ticker = tuned.ticker AND base.date = tuned.date
                JOIN feat_targets t 
                    ON base.ticker = t.ticker AND base.date = t.date
                WHERE t.ret_5d_f IS NOT NULL
            """).fetchone()
            
            logger.info(f"\n  Comparison (base v2 vs tuned v2):")
            logger.info(f"    Base IC:  {comparison[0]:.4f}")
            logger.info(f"    Tuned IC: {comparison[1]:.4f}")
            logger.info(f"    Correlation between models: {comparison[2]:.4f}")
            
            if comparison[0] != 0:
                improvement = (comparison[1] - comparison[0]) / abs(comparison[0]) * 100
                logger.info(f"    Improvement: {improvement:+.1f}%")
    except Exception as e:
        logger.warning(f"  Could not compare to base v2: {e}")
    
    # Check ticker/date coverage
    coverage = con.execute("""
        SELECT 
            (SELECT COUNT(DISTINCT date) FROM feat_alpha_ml_xgb_v2_tuned) as ml_dates,
            (SELECT COUNT(DISTINCT ticker) FROM feat_alpha_ml_xgb_v2_tuned) as ml_tickers
    """).fetchone()
    
    logger.info(f"  Coverage: {coverage[0]:,} dates, {coverage[1]:,} tickers")


def main():
    parser = argparse.ArgumentParser(description="Generate ML v2 TUNED predictions using Optuna-optimized models")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--model-dir", default="scripts/ml/outputs", help="Directory containing trained models")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    logger.info("=" * 70)
    logger.info("ML v2 TUNED PREDICTION GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Features: {len(FEATURES)}")
    logger.info("")
    logger.info("Using Optuna-tuned hyperparameters:")
    logger.info("  learning_rate:    0.0996")
    logger.info("  max_depth:        4")
    logger.info("  subsample:        0.545")
    logger.info("  colsample_bytree: 0.857")
    logger.info("  reg_lambda:       0.50")
    logger.info("  reg_alpha:        0.644")
    logger.info("  min_child_weight: 408")
    
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
        
        # Validate
        validate_predictions(con)
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)
        logger.info("Created table: feat_alpha_ml_xgb_v2_tuned")
        logger.info("  - alpha_ml_v2_tuned_reg (regression predictions)")
        logger.info("  - alpha_ml_v2_tuned_clf (classification probabilities)")
        logger.info("")
        logger.info("Matrix builder will join this table into feat_matrix_v2")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()