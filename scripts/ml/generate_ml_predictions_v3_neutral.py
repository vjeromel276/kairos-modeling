#!/usr/bin/env python3
"""
generate_ml_predictions_v3_neutral.py
======================================
Generate ML v3 SECTOR NEUTRAL predictions using sector-neutralized models.

This script runs AFTER train_xgb_alpha_v3_neutral.py has been run at least once
to produce the model files, and BEFORE the matrix builder runs.

It loads pre-trained XGBoost models (trained on sector-neutral target) and
generates predictions for ALL ticker/date pairs in sep_base_academic.

Creates: feat_alpha_ml_xgb_v3_neutral table
    - alpha_ml_v3_neutral_reg (regression predictions)
    - alpha_ml_v3_neutral_clf (classification probabilities) ← drives rebalance

Walk-forward CV results (from train_xgb_alpha_v3_neutral.py):
    - Mean IC (clf): 0.0281 vs 0.0244 baseline (+15%)
    - IC Sharpe:     1.479  vs 0.717  baseline (+106%)
    - % Positive:    100%   vs 86%    baseline

Usage:
    python scripts/ml/generate_ml_predictions_v3_neutral.py --db data/kairos.duckdb

Integration:
    Add to run_pipeline.py Phase 6, runs after generate_ml_predictions_v2_tuned.py
    Matrix builder (build_feature_matrix_v2.py) joins this table into feat_matrix_v2
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
# FEATURE CONFIGURATION — must match train_xgb_alpha_v3_neutral.py exactly
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

FEATURES = []
for cols in FEATURE_SOURCES.values():
    FEATURES.extend(cols)


# =============================================================================
# LOAD MODELS
# =============================================================================

def load_models(model_dir: Path):
    """Load trained v3 neutral models and feature medians."""

    logger.info(f"Loading v3 neutral models from {model_dir}...")

    # Regression model
    reg_path = model_dir / 'model_regression_v3_neutral.json'
    if not reg_path.exists():
        raise FileNotFoundError(
            f"Regression model not found: {reg_path}\n"
            f"Run train_xgb_alpha_v3_neutral.py first."
        )
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(reg_path))
    logger.info(f"  Loaded regression model: {reg_path}")

    # Classification model
    clf_path = model_dir / 'model_classification_v3_neutral.json'
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Classification model not found: {clf_path}\n"
            f"Run train_xgb_alpha_v3_neutral.py first."
        )
    model_clf = xgb.XGBClassifier()
    model_clf.load_model(str(clf_path))
    logger.info(f"  Loaded classification model: {clf_path}")

    # Feature medians
    medians_path = model_dir / 'feature_medians_v3_neutral.json'
    if not medians_path.exists():
        raise FileNotFoundError(
            f"Feature medians not found: {medians_path}\n"
            f"Run train_xgb_alpha_v3_neutral.py first."
        )
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    logger.info(f"  Loaded feature medians: {medians_path}")

    return model_reg, model_clf, medians


# =============================================================================
# LOAD DATA
# =============================================================================

def load_prediction_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load features for ALL ticker/date pairs in sep_base_academic.

    Uses sep_base_academic as the base (all ticker/date pairs we care about),
    then left joins each feature table.

    Note: We do NOT need to compute the sector-neutral target here.
    The models were trained on the neutral target, but at inference time
    we just need the features. The model outputs its prediction directly.
    """

    logger.info("Loading ticker/date pairs from sep_base_academic...")

    base_query = """
        SELECT DISTINCT ticker, date
        FROM sep_base_academic
        WHERE date >= '2015-01-01'
    """
    df = con.execute(base_query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Base grid: {len(df):,} rows")

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

        for col in cols:
            cov = df[col].notna().mean() * 100
            logger.info(f"    {col}: {cov:.1f}%")

    # Verify no duplicates introduced by joins
    dupes = df.duplicated(subset=['ticker', 'date']).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate ticker/date rows after feature joins: {dupes}")

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")

    return df


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(
    df: pd.DataFrame,
    model_reg: xgb.XGBRegressor,
    model_clf: xgb.XGBClassifier,
    medians: dict
) -> pd.DataFrame:
    """
    Generate predictions for all rows using v3 neutral models.

    The models were trained on sector-neutral targets but predictions
    are applied to the full cross-section. At inference time the model
    outputs a probability (clf) or return estimate (reg) for each stock.
    Ranking by clf probability gives the rebalance signal.
    """

    logger.info("Generating v3 neutral predictions...")

    X = df[FEATURES].copy()

    # Fill missing with training medians
    for col in FEATURES:
        median_val = medians.get(col, 0)
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            logger.debug(f"  Filling {n_missing:,} missing in {col} with {median_val:.4f}")
        X[col] = X[col].fillna(median_val)

    # Regression predictions
    logger.info("  Running regression model...")
    df['alpha_ml_v3_neutral_reg'] = model_reg.predict(X)

    # Classification predictions
    logger.info("  Running classification model...")
    df['alpha_ml_v3_neutral_clf'] = model_clf.predict_proba(X)[:, 1]

    # Summary stats
    logger.info(f"\nPrediction summary:")
    logger.info(f"  alpha_ml_v3_neutral_reg: "
                f"mean={df['alpha_ml_v3_neutral_reg'].mean():.4f}, "
                f"std={df['alpha_ml_v3_neutral_reg'].std():.4f}")
    logger.info(f"  alpha_ml_v3_neutral_clf: "
                f"mean={df['alpha_ml_v3_neutral_clf'].mean():.4f}, "
                f"std={df['alpha_ml_v3_neutral_clf'].std():.4f}")

    return df[['ticker', 'date', 'alpha_ml_v3_neutral_reg', 'alpha_ml_v3_neutral_clf']]


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(con: duckdb.DuckDBPyConnection, predictions_df: pd.DataFrame):
    """
    Save predictions to feat_alpha_ml_xgb_v3_neutral table.
    Existing v2_tuned table is never touched.
    """

    logger.info("Saving to feat_alpha_ml_xgb_v3_neutral...")

    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb_v3_neutral")
    con.register("predictions_df", predictions_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb_v3_neutral AS
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM feat_alpha_ml_xgb_v3_neutral"
    ).fetchone()[0]
    logger.info(f"  Created feat_alpha_ml_xgb_v3_neutral with {count:,} rows")

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_ml_v3_neutral_ticker_date
        ON feat_alpha_ml_xgb_v3_neutral(ticker, date)
    """)

    date_range = con.execute("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM feat_alpha_ml_xgb_v3_neutral
    """).fetchone()
    logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")


# =============================================================================
# VALIDATE
# =============================================================================

def validate_predictions(con: duckdb.DuckDBPyConnection):
    """
    Validate v3 neutral predictions and compare to v2_tuned.

    IC is always computed against raw ret_5d_f — what actually
    happens in the market regardless of what we trained on.
    """

    logger.info("\nValidating predictions...")

    # IC against raw returns
    try:
        result = con.execute("""
            SELECT
                CORR(ml.alpha_ml_v3_neutral_clf, t.ret_5d_f) as ic_clf,
                CORR(ml.alpha_ml_v3_neutral_reg, t.ret_5d_f) as ic_reg,
                COUNT(*) as n
            FROM feat_alpha_ml_xgb_v3_neutral ml
            JOIN feat_targets t
                ON ml.ticker = t.ticker AND ml.date = t.date
            WHERE t.ret_5d_f IS NOT NULL
        """).fetchone()

        logger.info(f"  IC (v3 neutral clf vs ret_5d_f): {result[0]:.4f}")
        logger.info(f"  IC (v3 neutral reg vs ret_5d_f): {result[1]:.4f}")
        logger.info(f"  (n = {result[2]:,})")

    except Exception as e:
        logger.warning(f"  Could not compute IC: {e}")

    # Compare to v2_tuned if available
    try:
        v2_exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'feat_alpha_ml_xgb_v2_tuned'
        """).fetchone()[0] > 0

        if v2_exists:
            comparison = con.execute("""
                SELECT
                    CORR(v2.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_v2_tuned,
                    CORR(v3.alpha_ml_v3_neutral_clf, t.ret_5d_f) as ic_v3_neutral,
                    CORR(v2.alpha_ml_v2_tuned_clf,
                         v3.alpha_ml_v3_neutral_clf) as correlation,
                    COUNT(*) as n
                FROM feat_alpha_ml_xgb_v2_tuned v2
                JOIN feat_alpha_ml_xgb_v3_neutral v3
                    ON v2.ticker = v3.ticker AND v2.date = v3.date
                JOIN feat_targets t
                    ON v2.ticker = t.ticker AND v2.date = t.date
                WHERE t.ret_5d_f IS NOT NULL
            """).fetchone()

            logger.info(f"\n  Comparison vs v2_tuned:")
            logger.info(f"    v2_tuned IC:    {comparison[0]:.4f}")
            logger.info(f"    v3_neutral IC:  {comparison[1]:.4f}")
            logger.info(f"    Correlation between signals: {comparison[2]:.4f}")

            if comparison[0] and comparison[0] != 0:
                improvement = (
                    (comparison[1] - comparison[0]) / abs(comparison[0]) * 100
                )
                logger.info(f"    Improvement: {improvement:+.1f}%")
        else:
            logger.info("  v2_tuned table not found — skipping comparison")

    except Exception as e:
        logger.warning(f"  Could not compare to v2_tuned: {e}")

    # Coverage check
    coverage = con.execute("""
        SELECT
            COUNT(DISTINCT date)   as dates,
            COUNT(DISTINCT ticker) as tickers
        FROM feat_alpha_ml_xgb_v3_neutral
    """).fetchone()
    logger.info(f"\n  Coverage: {coverage[0]:,} dates, {coverage[1]:,} tickers")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ML v3 neutral predictions"
    )
    parser.add_argument("--db",        required=True, help="Path to DuckDB database")
    parser.add_argument("--model-dir", default="scripts/ml/outputs",
                        help="Directory containing trained v3 neutral models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    logger.info("=" * 70)
    logger.info("ML v3 SECTOR NEUTRAL PREDICTION GENERATION")
    logger.info("=" * 70)
    logger.info(f"Database:  {args.db}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Features:  {len(FEATURES)}")
    logger.info("")
    logger.info("Expected model files:")
    logger.info(f"  {model_dir}/model_regression_v3_neutral.json")
    logger.info(f"  {model_dir}/model_classification_v3_neutral.json")
    logger.info(f"  {model_dir}/feature_medians_v3_neutral.json")

    # Load models
    model_reg, model_clf, medians = load_models(model_dir)

    # Connect — needs write access to create the output table
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
        logger.info("Created table: feat_alpha_ml_xgb_v3_neutral")
        logger.info("  - alpha_ml_v3_neutral_reg (regression predictions)")
        logger.info("  - alpha_ml_v3_neutral_clf (classification probabilities)")
        logger.info("")
        logger.info("alpha_ml_v3_neutral_clf drives the weekly rebalance")
        logger.info("Matrix builder will join this table into feat_matrix_v2")

    finally:
        con.close()


if __name__ == "__main__":
    main()