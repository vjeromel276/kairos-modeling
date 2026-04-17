#!/usr/bin/env python3
"""
generate_ml_predictions_v3_neutral.py
======================================
Generate ML v3 SECTOR NEUTRAL predictions using the proven CPCV-validated model.

This script loads the proven XGBoost classifier (trained on sector-neutral target,
n_estimators=100, no early stopping) and generates predictions for ALL ticker/date
pairs in sep_base_academic.

Model: xgb_v3_neutral_n100_20260412.joblib (classification only)
    - CPCV mean IC: 0.0259
    - CPCV IC Sharpe: 3.727
    - 100% positive folds (15/15)
    - Target: ret_5d_sector_neutral (training only; inference uses raw features)

Creates: feat_alpha_ml_xgb_v3_neutral table
    - alpha_ml_v3_neutral_clf (classification probabilities) ← drives rebalance

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
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# PROVEN MODEL ARTIFACT PATHS
# =============================================================================

DEFAULT_MODEL_PATH = (
    "/mnt/DATA01/media/vjerome2/Extreme Pro/ml_experiments/models/"
    "xgb_v3_neutral_n100_20260412.joblib"
)
DEFAULT_FEATURES_PATH = (
    "/mnt/DATA01/media/vjerome2/Extreme Pro/ml_experiments/models/"
    "xgb_v3_neutral_n100_20260412_features.txt"
)

# =============================================================================
# FEATURE CONFIGURATION — loaded from model artifact, with fallback
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


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(model_path: str, features_path: str, medians_path: str):
    """Load proven v3 neutral classifier, feature list, and medians."""

    logger.info(f"Loading proven v3 neutral model...")

    # Classification model (joblib)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Expected the proven CPCV-validated joblib model."
        )
    model_clf = joblib.load(model_path)
    logger.info(f"  Loaded classifier: {model_path}")
    logger.info(f"  Model type: {type(model_clf).__name__}")

    # Feature list
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list not found: {features_path}")
    features = [f.strip() for f in features_path.read_text().strip().split('\n') if f.strip()]
    logger.info(f"  Features: {len(features)} loaded from {features_path}")

    # Feature medians for missing value imputation
    medians_path = Path(medians_path)
    if not medians_path.exists():
        raise FileNotFoundError(f"Feature medians not found: {medians_path}")
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    logger.info(f"  Loaded feature medians: {medians_path}")

    return model_clf, features, medians


# =============================================================================
# LOAD DATA
# =============================================================================

def load_prediction_data(con: duckdb.DuckDBPyConnection, features: list) -> pd.DataFrame:
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
        # Only join columns that are in the model's feature list
        needed_cols = [c for c in cols if c in features]
        if not needed_cols:
            continue

        logger.info(f"  Joining {table}...")

        cols_str = ', '.join(needed_cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])

        df = df.merge(feat_df, on=['ticker', 'date'], how='left')

        for col in needed_cols:
            cov = df[col].notna().mean() * 100
            logger.info(f"    {col}: {cov:.1f}%")

    # Verify no duplicates introduced by joins
    dupes = df.duplicated(subset=['ticker', 'date']).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate ticker/date rows after feature joins: {dupes}")

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(features)} features")

    return df


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(
    df: pd.DataFrame,
    model_clf,
    features: list,
    medians: dict
) -> pd.DataFrame:
    """
    Generate predictions for all rows using proven v3 neutral classifier.

    The model was trained on sector-neutral target but predictions
    are applied to the full cross-section. At inference time the model
    outputs a probability for each stock. Ranking by probability gives
    the rebalance signal.
    """

    logger.info("Generating v3 neutral predictions...")

    X = df[features].copy()

    # Fill missing with training medians
    for col in features:
        median_val = medians.get(col, 0)
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            logger.debug(f"  Filling {n_missing:,} missing in {col} with {median_val:.4f}")
        X[col] = X[col].fillna(median_val)

    # Classification predictions
    logger.info("  Running classification model...")
    df['alpha_ml_v3_neutral_clf'] = model_clf.predict_proba(X)[:, 1]

    # Summary stats
    logger.info(f"\nPrediction summary:")
    logger.info(f"  alpha_ml_v3_neutral_clf: "
                f"mean={df['alpha_ml_v3_neutral_clf'].mean():.4f}, "
                f"std={df['alpha_ml_v3_neutral_clf'].std():.4f}")

    # Sanity checks per research plan
    pred_std = df['alpha_ml_v3_neutral_clf'].std()
    pred_mean = df['alpha_ml_v3_neutral_clf'].mean()
    if pred_std < 0.005:
        logger.warning(f"  WARNING: pred_std={pred_std:.4f} — model may be degenerate")
    if abs(pred_mean - 0.5) > 0.1:
        logger.warning(f"  WARNING: pred_mean={pred_mean:.4f} — far from 0.5, check model")

    return df[['ticker', 'date', 'alpha_ml_v3_neutral_clf']]


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
                COUNT(*) as n
            FROM feat_alpha_ml_xgb_v3_neutral ml
            JOIN feat_targets t
                ON ml.ticker = t.ticker AND ml.date = t.date
            WHERE t.ret_5d_f IS NOT NULL
        """).fetchone()

        logger.info(f"  IC (v3 neutral clf vs ret_5d_f): {result[0]:.4f}")
        logger.info(f"  (n = {result[1]:,})")

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
        description="Generate ML v3 neutral predictions using proven CPCV model"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to proven joblib model")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH,
                        help="Path to feature list file")
    parser.add_argument("--medians", default="scripts/ml/outputs/feature_medians_v3_neutral.json",
                        help="Path to feature medians JSON")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ML v3 SECTOR NEUTRAL PREDICTION GENERATION")
    logger.info("  Proven CPCV model: IC Sharpe 3.727, 100% positive folds")
    logger.info("=" * 70)
    logger.info(f"Database:  {args.db}")
    logger.info(f"Model:     {args.model}")
    logger.info(f"Features:  {args.features}")
    logger.info(f"Medians:   {args.medians}")

    # Load model
    model_clf, features, medians = load_model(args.model, args.features, args.medians)

    # Connect — needs write access to create the output table
    con = duckdb.connect(args.db)

    try:
        # Load data
        df = load_prediction_data(con, features)

        # Generate predictions
        predictions_df = generate_predictions(df, model_clf, features, medians)

        # Save to database
        save_predictions(con, predictions_df)

        # Validate
        validate_predictions(con)

        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)
        logger.info("Created table: feat_alpha_ml_xgb_v3_neutral")
        logger.info("  - alpha_ml_v3_neutral_clf (classification probabilities)")
        logger.info("")
        logger.info("alpha_ml_v3_neutral_clf drives the weekly rebalance")
        logger.info("Matrix builder will join this table into feat_matrix_v2")

    finally:
        con.close()


if __name__ == "__main__":
    main()