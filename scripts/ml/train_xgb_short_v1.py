#!/usr/bin/env python3
"""
train_xgb_short_v1.py
======================
Train a SEPARATE short-specific XGBoost model to identify stocks that will
underperform their sector peers.

This is fundamentally different from inverting the long signal:
    - Target: is_loser = bottom quintile of sector-neutral 5-day returns
    - Feature profile: driven by quality/distress (high vol, low profitability,
      high beta) rather than momentum/value
    - Evaluated on: ability to predict the BOTTOM, not the top
    - Intended use: short book in bear/high-vol regimes only

Model design decisions:
    - Classification: predict P(loser) directly
    - Same 23 features as long model (proven feature set)
    - n_estimators=100, no early stopping (locked from CPCV research)
    - Sector-neutral target (remove sector rotation noise)
    - Walk-forward CV with Spearman IC against raw ret_5d_f

Does NOT touch any production tables or the long model.

Usage:
    python scripts/ml/train_xgb_short_v1.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_short_v1.py --db data/kairos.duckdb --skip-cv
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
# FEATURES — same 23 as long model
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
# TARGET: is_loser (bottom quintile of sector-neutral returns)
# =============================================================================

EVAL_TARGET = 'ret_5d_f'  # IC always evaluated against raw return

# =============================================================================
# LOCKED HYPERPARAMETERS (same structure as long model)
# =============================================================================

XGB_PARAMS = {
    'n_estimators':     100,
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
}

CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load features and build the short-specific target.

    Target: is_loser = 1 if stock is in the bottom quintile of
    sector-neutral 5-day forward returns for that date.
    """

    logger.info("Loading data for short model...")

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
    logger.info(f"  Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

    # Sector-neutral returns
    sector_mean = df.groupby(['date', 'sector'])['ret_5d_f'].transform('mean')
    df['ret_neutral'] = df['ret_5d_f'] - sector_mean

    # Target: bottom quintile = loser
    df['is_loser'] = df.groupby('date')['ret_neutral'].transform(
        lambda x: (pd.qcut(x, 5, labels=False, duplicates='drop') == 0).astype(int)
    )

    logger.info(f"  Loser rate: {df['is_loser'].mean():.1%} (expect ~20%)")

    # Join features
    for table, cols in FEATURE_SOURCES.items():
        logger.info(f"  Joining {table}...")
        cols_str = ', '.join(cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        df = df.merge(feat_df, on=['ticker', 'date'], how='left')

    df['year'] = df['date'].dt.year

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")

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


def calculate_short_metrics(y_true_ret: np.ndarray, y_pred_loser_prob: np.ndarray,
                            n_buckets: int = 10):
    """
    Evaluate short model by checking if high P(loser) stocks actually
    have negative forward returns.

    Returns metrics for the top prediction bucket (most likely losers).
    """
    mask = ~(np.isnan(y_true_ret) | np.isnan(y_pred_loser_prob))
    rets = y_true_ret[mask]
    preds = y_pred_loser_prob[mask]

    # Rank into buckets by predicted loser probability
    buckets = pd.qcut(preds, n_buckets, labels=False, duplicates='drop')

    # Top bucket = stocks most likely to be losers
    top_bucket_mask = buckets == buckets.max()
    top_rets = rets[top_bucket_mask]

    # Bottom bucket = stocks least likely to be losers (should do well)
    bottom_bucket_mask = buckets == 0
    bottom_rets = rets[bottom_bucket_mask]

    return {
        'top_bucket_mean_ret': float(top_rets.mean()),
        'top_bucket_win_rate': float((top_rets > 0).mean()),
        'bottom_bucket_mean_ret': float(bottom_rets.mean()),
        'spread': float(bottom_rets.mean() - top_rets.mean()),
        'ic_vs_ret': float(calculate_ic(rets, -preds)),  # negative: high P(loser) = low return
    }


# =============================================================================
# WALK-FORWARD CV
# =============================================================================

def run_walk_forward_cv(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Walk-forward CV for the short model."""

    results = []

    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD CV (SHORT MODEL v1)")
    logger.info("=" * 70)

    for test_year in CV_TEST_YEARS:
        train_df = df[df['year'] < test_year].copy()
        test_df = df[df['year'] == test_year].copy()

        if len(train_df) < 1000:
            continue

        # Purge
        train_dates = sorted(train_df['date'].unique())
        if len(train_dates) > PURGE_DAYS:
            purge_cutoff = train_dates[-(PURGE_DAYS + 1)]
            train_df = train_df[train_df['date'] <= purge_cutoff]

        logger.info(f"\n  {test_year}: train={len(train_df):,}  test={len(test_df):,}")

        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        y_train = train_df['is_loser'].values
        y_test_ret = test_df[EVAL_TARGET].values

        # Fill missing
        for col in FEATURES:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

        # Train
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict_proba(X_test)[:, 1]

        # Standard IC (high P(loser) should correlate with low returns)
        ic_neg = calculate_ic(y_test_ret, -y_pred)

        # Short-specific metrics
        short_metrics = calculate_short_metrics(y_test_ret, y_pred)

        logger.info(f"  {test_year}: IC={ic_neg:+.4f}  "
                    f"TopBucket(losers)={short_metrics['top_bucket_mean_ret']:+.4f}  "
                    f"Spread={short_metrics['spread']:+.4f}")

        results.append({
            'test_year': test_year,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'ic': ic_neg,
            'top_bucket_ret': short_metrics['top_bucket_mean_ret'],
            'bottom_bucket_ret': short_metrics['bottom_bucket_mean_ret'],
            'spread': short_metrics['spread'],
            'top_bucket_win_rate': short_metrics['top_bucket_win_rate'],
        })

    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results_short_v1.csv'
    results_df.to_csv(results_path, index=False)

    mean_ic = results_df['ic'].mean()
    std_ic = results_df['ic'].std()
    ic_sharpe = mean_ic / std_ic if std_ic > 0 else 0
    pct_positive = (results_df['ic'] > 0).mean()
    mean_spread = results_df['spread'].mean()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SHORT MODEL CV SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Mean IC (neg pred vs ret):  {mean_ic:.4f}")
    logger.info(f"  Std IC:                     {std_ic:.4f}")
    logger.info(f"  IC Sharpe:                  {ic_sharpe:.3f}")
    logger.info(f"  % Positive IC:              {pct_positive:.0%}")
    logger.info(f"  Mean spread (bottom-top):   {mean_spread:+.4f}")
    logger.info(f"  Mean top bucket return:     {results_df['top_bucket_ret'].mean():+.4f}")
    logger.info(f"  Mean top bucket win rate:   {results_df['top_bucket_win_rate'].mean():.1%}")
    logger.info(f"\nPer-fold:")
    for _, row in results_df.iterrows():
        logger.info(f"  {int(row['test_year'])}: IC={row['ic']:+.4f}  "
                    f"TopRet={row['top_bucket_ret']:+.4f}  "
                    f"Spread={row['spread']:+.4f}")

    return results_df


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df: pd.DataFrame, output_dir: Path):
    """Train final short classifier on all data."""

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL SHORT MODEL (all data, n=100)")
    logger.info("=" * 70)

    X = df[FEATURES].copy()
    y = df['is_loser'].values

    # Compute and save feature medians
    medians = {}
    for col in FEATURES:
        medians[col] = float(X[col].median())
        X[col] = X[col].fillna(medians[col])

    medians_path = output_dir / 'feature_medians_short_v1.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    logger.info(f"  Saved feature medians: {medians_path}")

    # Train
    logger.info(f"  Training on {len(X):,} rows...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    # Save
    today = datetime.now().strftime('%Y%m%d')
    model_path = output_dir / f'xgb_short_v1_{today}.joblib'
    joblib.dump(model, model_path)
    logger.info(f"  Saved model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    features_path = output_dir / f'xgb_short_v1_{today}_features.txt'
    features_path.write_text('\n'.join(FEATURES) + '\n')
    logger.info(f"  Saved feature list: {features_path}")

    # Sanity check
    y_pred = model.predict_proba(X)[:, 1]
    logger.info(f"  pred_mean: {y_pred.mean():.4f} (expect ~0.20 for quintile target)")
    logger.info(f"  pred_std:  {y_pred.std():.4f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    logger.info(f"\n  Top 10 features by importance:")
    for feat, imp in importance.head(10).items():
        logger.info(f"    {feat:<20} {imp:.4f}")

    return model, medians


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train separate short-specific XGBoost model"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", default="scripts/ml/outputs")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SHORT MODEL v1 — Separate loser prediction")
    logger.info("  Target: bottom quintile of sector-neutral ret_5d")
    logger.info("  Purpose: short book in bear/high-vol regimes")
    logger.info("=" * 70)

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


if __name__ == "__main__":
    main()
