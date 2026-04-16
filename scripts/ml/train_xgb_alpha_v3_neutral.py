#!/usr/bin/env python3
"""
train_xgb_alpha_v3_neutral.py
==============================
XGBoost ML v3 with sector-neutralized training target.

Changes from v2_tuned:
- Training target changed from ret_5d_f to ret_5d_sector_neutral
- ret_5d_sector_neutral = ret_5d_f - equal_weight_sector_mean_ret_5d
- Classification label derived from ret_5d_sector_neutral > 0
- IC is always evaluated against raw ret_5d_f (what happens in the market)
- All hyperparameters identical to v2_tuned (Optuna Trial 78)

Rationale:
- Sector rotation explains ~20-30% of weekly stock return variance
- Model has no edge predicting sector rotation
- Removing it from the target lets the model focus on idiosyncratic return
- Walk-forward CV showed: Mean IC +15%, IC Sharpe 0.72 -> 1.60, 100% positive folds

Usage:
    python scripts/ml/train_xgb_alpha_v3_neutral.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_alpha_v3_neutral.py --db data/kairos.duckdb --skip-cv
"""

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURES — identical to v2_tuned
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
# TARGETS
# =============================================================================

TARGET_REG = 'ret_5d_sector_neutral'   # what we train on
TARGET_CLF = 'label_5d_sector_neutral' # derived from TARGET_REG > 0
EVAL_TARGET = 'ret_5d_f'              # always evaluate IC against raw return

# =============================================================================
# WALK-FORWARD CV
# =============================================================================

CV_START_YEAR = 2015
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5

# =============================================================================
# HYPERPARAMETERS — identical to v2_tuned (Optuna Trial 78)
# =============================================================================

XGB_PARAMS_REG = {
    'learning_rate':    0.0996,
    'max_depth':        4,
    'subsample':        0.545,
    'colsample_bytree': 0.857,
    'reg_lambda':       0.50,
    'reg_alpha':        0.644,
    'min_child_weight': 408,
    'objective':        'reg:squarederror',
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
}

XGB_PARAMS_CLF = {
    'learning_rate':    0.0996,
    'max_depth':        4,
    'subsample':        0.545,
    'colsample_bytree': 0.857,
    'reg_lambda':       0.50,
    'reg_alpha':        0.644,
    'min_child_weight': 408,
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
}

N_ESTIMATORS          = 500
EARLY_STOPPING_ROUNDS = 50


# =============================================================================
# DATA LOADING
# =============================================================================

def build_sector_neutral_target(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build ret_5d_sector_neutral from feat_targets and tickers.

    Method:
        1. Load ret_5d_f from feat_targets
        2. Join sector from tickers (DISTINCT to avoid duplicate table rows)
        3. Compute equal-weight sector mean return per date
        4. Subtract: ret_5d_sector_neutral = ret_5d_f - sector_mean_ret

    This is the variant proven to improve IC in notebook path1_neutral_target.ipynb.
    Walk-forward CV result: mean IC 0.0281 vs 0.0244 baseline, IC Sharpe 1.60 vs 0.72.
    """

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
        ) tk ON t.ticker = tk.ticker
        WHERE t.ret_5d_f IS NOT NULL
          AND t.date >= '2015-01-01'
        ORDER BY t.date, t.ticker
    """).fetchdf()

    df['date'] = pd.to_datetime(df['date'])

    # Verify no duplicates from the join
    dupes = df.duplicated(subset=['ticker', 'date']).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate ticker/date rows after join: {dupes}. "
                         f"Check tickers table for multiple sectors per ticker.")

    logger.info(f"  Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Sector coverage: {df['sector'].notna().mean():.1%}")

    # Equal-weight sector return per date
    sector_ret = (
        df.groupby(['date', 'sector'])['ret_5d_f']
        .mean()
        .rename('sector_ret_5d')
    )
    df = df.join(sector_ret, on=['date', 'sector'])

    # Sector-neutral target
    df['ret_5d_sector_neutral'] = df['ret_5d_f'] - df['sector_ret_5d']

    # Validate: cross-sectional mean should be exactly zero
    # Validate: cross-sectional std should be lower than raw
    mean_neutral = df.groupby('date')['ret_5d_sector_neutral'].mean().mean()
    std_raw      = df.groupby('date')['ret_5d_f'].std().mean()
    std_neutral  = df.groupby('date')['ret_5d_sector_neutral'].std().mean()

    logger.info(f"  Validation:")
    logger.info(f"    Cross-sectional mean (should be ~0): {mean_neutral:.6f}")
    logger.info(f"    Cross-sectional std raw:     {std_raw:.4f}")
    logger.info(f"    Cross-sectional std neutral: {std_neutral:.4f}")

    if abs(mean_neutral) > 0.001:
        raise ValueError(f"Sector neutral mean too large: {mean_neutral:.6f}. "
                         f"Something is wrong with the neutralization.")

    if std_neutral >= std_raw:
        raise ValueError(f"Sector neutral std ({std_neutral:.4f}) >= raw std ({std_raw:.4f}). "
                         f"Neutralization is not removing variance as expected.")

    return df[['ticker', 'date', 'ret_5d_f', 'ret_5d_sector_neutral']]


def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load sector-neutral targets and join all feature tables.
    Identical join logic to v2_tuned except base comes from
    build_sector_neutral_target() instead of feat_targets directly.
    """

    logger.info("Loading features...")

    # Start with neutralized targets
    df = build_sector_neutral_target(con)

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

    # Derive targets
    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    df['year']     = df['date'].dt.year

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")
    logger.info(f"  Target positive rate: {df[TARGET_CLF].mean():.1%} "
                f"(expect ~50% for well-neutralized target)")

    return df


# =============================================================================
# UTILITIES
# =============================================================================

def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and raw returns."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def get_train_test_split(df: pd.DataFrame, test_year: int, purge_days: int = 5):
    """Walk-forward split with purging — identical to v2_tuned."""

    test_mask  = df['year'] == test_year
    train_mask = df['year'] <  test_year

    train_dates = df.loc[train_mask, 'date']
    if len(train_dates) == 0:
        return None, None

    unique_train_dates = sorted(train_dates.unique())
    if len(unique_train_dates) <= purge_days:
        return None, None

    purge_cutoff = unique_train_dates[-(purge_days + 1)]
    train_mask_purged = (df['year'] < test_year) & (df['date'] <= purge_cutoff)

    return df.loc[train_mask_purged].copy(), df.loc[test_mask].copy()


# =============================================================================
# WALK-FORWARD CV
# =============================================================================

def run_walk_forward_cv(df: pd.DataFrame, output_dir: Path):
    """
    Walk-forward CV using sector-neutral target for training,
    raw ret_5d_f for IC evaluation.
    """

    results = []

    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD CV (v3 SECTOR NEUTRAL)")
    logger.info("=" * 70)

    for test_year in CV_TEST_YEARS:
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST YEAR: {test_year}")
        logger.info(f"{'='*50}")

        train_df, test_df = get_train_test_split(df, test_year, PURGE_DAYS)

        if train_df is None or len(train_df) < 1000:
            logger.warning(f"Insufficient training data for {test_year}")
            continue

        logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

        X_train = train_df[FEATURES].copy()
        X_test  = test_df[FEATURES].copy()

        # Training targets (sector neutral)
        y_train_reg = train_df[TARGET_REG].values
        y_train_clf = train_df[TARGET_CLF].values

        # Evaluation target (always raw — what actually happens in market)
        y_test_reg = test_df[EVAL_TARGET].values
        y_test_clf = (test_df[EVAL_TARGET] > 0).astype(int).values

        # Fill missing with training medians
        medians = {}
        for col in FEATURES:
            medians[col] = X_train[col].median()
            X_train[col] = X_train[col].fillna(medians[col])
            X_test[col]  = X_test[col].fillna(medians[col])

        # Validation split (last 10% of training)
        n_val = int(len(X_train) * 0.1)

        X_tr = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_tr_reg = y_train_reg[:-n_val]
        y_val_reg = y_train_reg[-n_val:]
        y_tr_clf = y_train_clf[:-n_val]
        y_val_clf = y_train_clf[-n_val:]

        # ── Regression ───────────────────────────────────────────────────
        logger.info("Training regression model...")

        model_reg = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_REG
        )
        model_reg.fit(
            X_tr, y_tr_reg,
            eval_set=[(X_val, y_val_reg)],
            verbose=False
        )

        y_pred_reg = model_reg.predict(X_test)
        ic_reg     = calculate_ic(y_test_reg, y_pred_reg)
        logger.info(f"Regression IC: {ic_reg:.4f} (trees: {model_reg.best_iteration})")

        # ── Classification ───────────────────────────────────────────────
        logger.info("Training classification model...")

        model_clf = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_CLF
        )
        model_clf.fit(
            X_tr, y_tr_clf,
            eval_set=[(X_val, y_val_clf)],
            verbose=False
        )

        y_pred_clf = model_clf.predict_proba(X_test)[:, 1]
        ic_clf     = calculate_ic(y_test_reg, y_pred_clf)
        auc_clf    = roc_auc_score(y_test_clf, y_pred_clf)

        logger.info(f"Classification IC: {ic_clf:.4f}, AUC: {auc_clf:.4f} "
                    f"(trees: {model_clf.best_iteration})")

        results.append({
            'test_year':         test_year,
            'n_train':           len(train_df),
            'n_test':            len(test_df),
            'ic_regression':     ic_reg,
            'ic_classification': ic_clf,
            'auc_classification': auc_clf,
            'n_trees_reg':       model_reg.best_iteration,
            'n_trees_clf':       model_clf.best_iteration,
        })

    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results_v3_neutral.csv'
    results_df.to_csv(results_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("CV SUMMARY (v3 SECTOR NEUTRAL)")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    logger.info(f"\nMean IC (Regression):     {results_df['ic_regression'].mean():.4f}")
    logger.info(f"Mean IC (Classification): {results_df['ic_classification'].mean():.4f}")
    logger.info(f"Std IC  (Classification): {results_df['ic_classification'].std():.4f}")
    logger.info(f"IC Sharpe (Classification): "
                f"{results_df['ic_classification'].mean() / results_df['ic_classification'].std():.4f}")
    logger.info(f"% Positive IC: "
                f"{(results_df['ic_classification'] > 0).mean():.0%}")

    return results_df


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_models(df: pd.DataFrame, output_dir: Path):
    """Train final models on all data using sector-neutral targets."""

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL MODELS ON ALL DATA (v3 SECTOR NEUTRAL)")
    logger.info("=" * 70)

    X    = df[FEATURES].copy()
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values

    # Compute and save feature medians from full dataset
    medians = {}
    for col in FEATURES:
        medians[col] = float(X[col].median())
        X[col] = X[col].fillna(medians[col])

    medians_path = output_dir / 'feature_medians_v3_neutral.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    logger.info(f"Saved feature medians: {medians_path}")

    n     = len(X)
    n_val = int(n * 0.1)

    X_train = X.iloc[:-n_val]
    X_val   = X.iloc[-n_val:]

    # ── Regression ───────────────────────────────────────────────────────
    logger.info(f"Training final regression model on {len(X_train):,} rows...")

    model_reg = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS_REG
    )
    model_reg.fit(
        X_train, y_reg[:-n_val],
        eval_set=[(X_val, y_reg[-n_val:])],
        verbose=False
    )

    reg_path = output_dir / 'model_regression_v3_neutral.json'
    model_reg.save_model(str(reg_path))
    logger.info(f"Saved: {reg_path} (trees: {model_reg.best_iteration})")

    # ── Classification ───────────────────────────────────────────────────
    logger.info("Training final classification model...")

    model_clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS_CLF
    )
    model_clf.fit(
        X_train, y_clf[:-n_val],
        eval_set=[(X_val, y_clf[-n_val:])],
        verbose=False
    )

    clf_path = output_dir / 'model_classification_v3_neutral.json'
    model_clf.save_model(str(clf_path))
    logger.info(f"Saved: {clf_path} (trees: {model_clf.best_iteration})")

    return model_reg, model_clf, medians


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost v3 with sector-neutralized target"
    )
    parser.add_argument("--db",         required=True, help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="scripts/ml/outputs", help="Output directory")
    parser.add_argument("--skip-cv",    action="store_true", help="Skip CV, only train final model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("XGBOOST ALPHA v3 SECTOR NEUTRAL")
    logger.info("=" * 70)
    logger.info(f"Database:   {args.db}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Features:   {len(FEATURES)}")
    logger.info(f"Target:     {TARGET_REG}")
    logger.info(f"Eval on:    {EVAL_TARGET} (raw return)")
    logger.info("")
    logger.info("Hyperparameters (identical to v2_tuned):")
    logger.info(f"  learning_rate:    {XGB_PARAMS_CLF['learning_rate']}")
    logger.info(f"  max_depth:        {XGB_PARAMS_CLF['max_depth']}")
    logger.info(f"  subsample:        {XGB_PARAMS_CLF['subsample']}")
    logger.info(f"  colsample_bytree: {XGB_PARAMS_CLF['colsample_bytree']}")
    logger.info(f"  reg_lambda:       {XGB_PARAMS_CLF['reg_lambda']}")
    logger.info(f"  reg_alpha:        {XGB_PARAMS_CLF['reg_alpha']}")
    logger.info(f"  min_child_weight: {XGB_PARAMS_CLF['min_child_weight']}")

    con = duckdb.connect(args.db)

    try:
        df = load_data(con)

        if not args.skip_cv:
            cv_results = run_walk_forward_cv(df, output_dir)

        model_reg, model_clf, medians = train_final_models(df, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Outputs saved to: {output_dir}")
        logger.info("  - cv_results_v3_neutral.csv")
        logger.info("  - model_regression_v3_neutral.json")
        logger.info("  - model_classification_v3_neutral.json")
        logger.info("  - feature_medians_v3_neutral.json")
        logger.info("")
        logger.info("Next step: run generate_ml_predictions_v3_neutral.py")

    finally:
        con.close()


if __name__ == "__main__":
    main()