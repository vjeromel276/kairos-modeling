#!/usr/bin/env python3
"""
train_xgb_alpha_v3_neutral_fixed.py
=====================================
Train XGBoost v3 sector-neutral classifier with CPCV-validated parameters.

Fixes from the broken train_xgb_alpha_v3_neutral.py:
    - Classification ONLY (no regression — proven model is clf-only)
    - n_estimators=100 FIXED (no early stopping — CPCV grid search result)
    - Locked hyperparameters from research plan (not Optuna v2_tuned params)
    - Saves as joblib (matches generate_ml_predictions_v3_neutral.py)
    - Walk-forward CV uses full train set, no val split (no early stopping)

Research evidence (kairos_ml_research_plan.md):
    - CPCV mean IC: 0.0259, IC Sharpe: 3.727, 100% positive folds
    - Early stopping is fundamentally incompatible with financial time series
    - n_estimators=100 fixed via CPCV grid search

Does NOT touch:
    - Any production tables
    - feat_alpha_ml_xgb_v2_tuned
    - feat_matrix_v2
    - Any existing model files

Usage:
    python scripts/ml/train_xgb_alpha_v3_neutral_fixed.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_alpha_v3_neutral_fixed.py --db data/kairos.duckdb --skip-cv
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURES — identical to v2_tuned (23 features)
# =============================================================================

FEATURE_SOURCES = {
    'feat_fundamental': [
        'earnings_yield', 'fcf_yield', 'roa', 'book_to_market',
        'operating_margin', 'roe',
    ],
    'feat_vol_sizing': [
        'vol_21', 'vol_63', 'vol_blend',
    ],
    'feat_beta': [
        'beta_21d', 'beta_63d', 'beta_252d', 'resid_vol_63d',
    ],
    'feat_price_action': [
        'hl_ratio', 'range_pct', 'ret_21d', 'ret_5d',
    ],
    'feat_momentum_v2': [
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1', 'reversal_1m',
    ],
}

FEATURES = []
for cols in FEATURE_SOURCES.values():
    FEATURES.extend(cols)

# =============================================================================
# TARGETS
# =============================================================================

TARGET_CLF = 'label_5d_sector_neutral'  # derived from ret_5d_sector_neutral > 0
EVAL_TARGET = 'ret_5d_f'               # IC always evaluated against raw return

# =============================================================================
# LOCKED HYPERPARAMETERS (CPCV-validated, from research plan)
# early_stopping: REMOVED PERMANENTLY
# =============================================================================

XGB_PARAMS = {
    'n_estimators':     100,        # FIXED — CPCV grid search
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.7,
    'colsample_bytree': 0.7,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'min_child_weight': 50,
    'objective':        'binary:logistic',
    'random_state':     42,
    'verbosity':        0,
    # early_stopping_rounds: REMOVED PERMANENTLY
}

# =============================================================================
# CV CONFIG
# =============================================================================

CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def build_sector_neutral_target(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build ret_5d_sector_neutral from feat_targets and tickers."""

    logger.info("Building sector-neutralized target...")

    df = con.execute("""
        SELECT
            t.ticker,
            t.date,
            t.ret_5d_f,
            tk.sector
        FROM feat_targets t
        LEFT JOIN (
            SELECT DISTINCT ticker, sector
            FROM tickers
            WHERE sector IS NOT NULL
              AND ticker != 'N/A'
        ) tk ON t.ticker = tk.ticker
        WHERE t.ret_5d_f IS NOT NULL
          AND t.date >= '2015-01-01'
        ORDER BY t.date, t.ticker
    """).fetchdf()

    df['date'] = pd.to_datetime(df['date'])

    dupes = df.duplicated(subset=['ticker', 'date']).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate ticker/date rows after join: {dupes}")

    logger.info(f"  Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Equal-weight sector return per date
    sector_ret = (
        df.groupby(['date', 'sector'])['ret_5d_f']
        .mean()
        .rename('sector_ret_5d')
    )
    df = df.join(sector_ret, on=['date', 'sector'])

    # Sector-neutral target
    df['ret_5d_sector_neutral'] = df['ret_5d_f'] - df['sector_ret_5d']

    # Validate
    mean_neutral = df.groupby('date')['ret_5d_sector_neutral'].mean().mean()
    logger.info(f"  Cross-sectional mean (should be ~0): {mean_neutral:.6f}")

    if abs(mean_neutral) > 0.001:
        raise ValueError(f"Sector neutral mean too large: {mean_neutral:.6f}")

    return df[['ticker', 'date', 'ret_5d_f', 'ret_5d_sector_neutral', 'sector']]


def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load sector-neutral targets and join all feature tables."""

    df = build_sector_neutral_target(con)

    for table, cols in FEATURE_SOURCES.items():
        logger.info(f"  Joining {table}...")
        cols_str = ', '.join(cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        df = df.merge(feat_df, on=['ticker', 'date'], how='left')

    # Derive classification label
    df[TARGET_CLF] = (df['ret_5d_sector_neutral'] > 0).astype(int)
    df['year'] = df['date'].dt.year

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")
    logger.info(f"  Positive rate: {df[TARGET_CLF].mean():.1%}")

    return df


# =============================================================================
# UTILITIES
# =============================================================================

def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


# =============================================================================
# WALK-FORWARD CV
# =============================================================================

def run_walk_forward_cv(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Walk-forward CV: classification only, no early stopping.
    Train on all years < test_year (no val split needed).
    IC evaluated against raw ret_5d_f.
    """

    results = []

    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD CV (v3 NEUTRAL FIXED — n=100, no early stopping)")
    logger.info("=" * 70)

    for test_year in CV_TEST_YEARS:
        # Split with purging
        train_df = df[df['year'] < test_year].copy()
        test_df = df[df['year'] == test_year].copy()

        if len(train_df) < 1000:
            logger.warning(f"Insufficient training data for {test_year}")
            continue

        # Purge last PURGE_DAYS of training
        train_dates = sorted(train_df['date'].unique())
        if len(train_dates) > PURGE_DAYS:
            purge_cutoff = train_dates[-(PURGE_DAYS + 1)]
            train_df = train_df[train_df['date'] <= purge_cutoff]

        logger.info(f"\n  {test_year}: train={len(train_df):,}  test={len(test_df):,}")

        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        y_train = train_df[TARGET_CLF].values
        y_test_raw = test_df[EVAL_TARGET].values

        # Fill missing with training medians
        for col in FEATURES:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

        # Train — no val split, no early stopping
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict_proba(X_test)[:, 1]
        ic = calculate_ic(y_test_raw, y_pred)

        logger.info(f"  {test_year}: IC={ic:+.4f}  (trees=100)")

        results.append({
            'test_year': test_year,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'ic': ic,
        })

    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results_v3_neutral_fixed.csv'
    results_df.to_csv(results_path, index=False)

    mean_ic = results_df['ic'].mean()
    std_ic = results_df['ic'].std()
    ic_sharpe = mean_ic / std_ic if std_ic > 0 else 0
    pct_positive = (results_df['ic'] > 0).mean()

    logger.info(f"\n{'=' * 50}")
    logger.info(f"CV SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Mean IC:      {mean_ic:.4f}")
    logger.info(f"  Std IC:       {std_ic:.4f}")
    logger.info(f"  IC Sharpe:    {ic_sharpe:.3f}")
    logger.info(f"  % Positive:   {pct_positive:.0%}")
    logger.info(f"\nPer-fold:")
    for _, row in results_df.iterrows():
        logger.info(f"  {int(row['test_year'])}: IC={row['ic']:+.4f}")

    return results_df


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df: pd.DataFrame, output_dir: Path):
    """Train final classifier on all data. No early stopping."""

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL MODEL (all data, n=100, no early stopping)")
    logger.info("=" * 70)

    X = df[FEATURES].copy()
    y = df[TARGET_CLF].values

    # Compute and save feature medians
    medians = {}
    for col in FEATURES:
        medians[col] = float(X[col].median())
        X[col] = X[col].fillna(medians[col])

    medians_path = output_dir / 'feature_medians_v3_neutral.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    logger.info(f"  Saved feature medians: {medians_path}")

    # Train on ALL data — no val split needed without early stopping
    logger.info(f"  Training on {len(X):,} rows...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    # Save as joblib (matches prediction script)
    today = datetime.now().strftime('%Y%m%d')
    model_path = output_dir / f'xgb_v3_neutral_n100_{today}.joblib'
    joblib.dump(model, model_path)
    logger.info(f"  Saved model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Save feature list
    features_path = output_dir / f'xgb_v3_neutral_n100_{today}_features.txt'
    features_path.write_text('\n'.join(FEATURES) + '\n')
    logger.info(f"  Saved feature list: {features_path}")

    # Sanity check predictions
    y_pred = model.predict_proba(X)[:, 1]
    logger.info(f"  pred_mean: {y_pred.mean():.4f} (expect ~0.51)")
    logger.info(f"  pred_std:  {y_pred.std():.4f} (expect >0.01)")

    # IC on training data (should be positive but not a valid OOS metric)
    ic_train = calculate_ic(df[EVAL_TARGET].values, y_pred)
    logger.info(f"  Train IC (not OOS): {ic_train:.4f}")

    return model, medians


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost v3 sector-neutral (fixed params, no early stopping)"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="scripts/ml/outputs",
                        help="Output directory for model artifacts")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip walk-forward CV, only train final model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("XGBOOST v3 SECTOR NEUTRAL — FIXED PARAMS")
    logger.info("  n_estimators=100, no early stopping (CPCV validated)")
    logger.info("=" * 70)
    logger.info(f"Database:   {args.db}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Features:   {len(FEATURES)}")
    logger.info(f"Params:     {XGB_PARAMS}")

    con = duckdb.connect(args.db, read_only=True)

    try:
        df = load_data(con)
    finally:
        con.close()

    if not args.skip_cv:
        run_walk_forward_cv(df, output_dir)

    model, medians = train_final_model(df, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs in: {output_dir}")
    logger.info("Next: run generate_ml_predictions_v3_neutral.py with the new model")


if __name__ == "__main__":
    main()
