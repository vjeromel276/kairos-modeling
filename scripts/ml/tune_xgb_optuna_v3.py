#!/usr/bin/env python3
"""
tune_xgb_optuna_v3.py
=====================
Hyperparameter tuning for XGBoost ML v3 using Optuna.

v3 uses 31 features (v2's 23 + 8 new ones). The hyperparameters tuned
for v2 (23 features) are not transferable — colsample_bytree in particular
behaves differently with a larger feature set. This script finds the
optimal parameters for the v3 feature space.

Features (31 total):
  v2 features (23): earnings_yield, fcf_yield, roa, book_to_market,
    operating_margin, roe, vol_21, vol_63, vol_blend, beta_21d, beta_63d,
    beta_252d, resid_vol_63d, hl_ratio, range_pct, ret_21d, ret_5d,
    mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1, reversal_1m
  New in v3 (8): asset_turnover, net_margin, debt_to_equity,
    price_vs_sma_21, sma_21_slope, macd_hist, insider_composite_z,
    gross_profitability

Search space:
  - learning_rate:    [0.005, 0.1]  log scale
  - max_depth:        [2, 6]
  - subsample:        [0.5, 0.9]
  - colsample_bytree: [0.5, 0.9]
  - reg_lambda:       [0.1, 10]     log scale
  - reg_alpha:        [0.0, 1.0]
  - min_child_weight: [50, 500]

Objective: Maximize mean classification IC across CV years (2019-2025)

Usage:
    # Run 50 trials
    python scripts/ml/tune_xgb_optuna_v3.py --db data/kairos.duckdb --n-trials 50

    # Quick smoke test with 5 trials
    python scripts/ml/tune_xgb_optuna_v3.py --db data/kairos.duckdb --n-trials 5

    # Resume a previous run
    python scripts/ml/tune_xgb_optuna_v3.py --db data/kairos.duckdb --n-trials 50 --resume

Output:
    - best_params_v3_tuned.json   Best hyperparameters found
    - optuna_study_v3.db          SQLite study (enables --resume)
    - tuning_results_v3.csv       All trials with scores
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# FEATURES — v3 set (31 features)
# =============================================================================

FEATURE_SOURCES = {
    'feat_fundamental': [
        'earnings_yield',
        'fcf_yield',
        'roa',
        'book_to_market',
        'operating_margin',
        'roe',
        'asset_turnover',       # new in v3
        'net_margin',           # new in v3
        'debt_to_equity',       # new in v3
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
    'feat_trend': [             # new in v3
        'price_vs_sma_21',
        'sma_21_slope',
        'macd_hist',
    ],
    'feat_insider': [           # new in v3
        'insider_composite_z',
    ],
    'feat_gross_profit': [      # new in v3
        'gross_profitability',
    ],
}

FEATURES = []
for cols in FEATURE_SOURCES.values():
    FEATURES.extend(cols)

TARGET_REG = 'ret_5d_f'
TARGET_CLF = 'label_5d_up'

# Walk-forward CV — matches train_xgb_alpha_v3.py exactly
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5

# Fixed params — not part of the search space
FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# BASELINE: v2_tuned params applied to 31 features
# This is what train_xgb_alpha_v3.py currently uses — our starting point.
# The tuner should beat this.
# =============================================================================

V2_TUNED_PARAMS = {
    'learning_rate': 0.0996,
    'max_depth': 4,
    'subsample': 0.545,
    'colsample_bytree': 0.857,
    'reg_lambda': 0.50,
    'reg_alpha': 0.644,
    'min_child_weight': 408,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(db_path: str) -> pd.DataFrame:
    """Load all 31 v3 features from the database."""

    logger.info("Loading data (31 features)...")
    con = duckdb.connect(db_path, read_only=True)

    # Base: ticker/date/target from feat_targets
    df = con.execute("""
        SELECT ticker, date, ret_5d_f
        FROM feat_targets
        WHERE date >= '2015-01-01'
          AND ret_5d_f IS NOT NULL
    """).fetchdf()
    df['date'] = pd.to_datetime(df['date'])

    # Join all feature tables
    for table, cols in FEATURE_SOURCES.items():
        cols_str = ', '.join(cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        df = df.merge(feat_df, on=['ticker', 'date'], how='left')

    con.close()

    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    df['year'] = df['date'].dt.year

    logger.info(f"Loaded {len(df):,} rows, {len(FEATURES)} features")

    # Log coverage for each feature
    logger.info("Feature coverage:")
    for feat in FEATURES:
        cov = df[feat].notna().mean() * 100
        logger.info(f"  {feat}: {cov:.1f}%")

    return df


# =============================================================================
# CV UTILITIES
# =============================================================================

def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman IC between predictions and realized returns."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def get_train_test_split(df: pd.DataFrame, test_year: int):
    """Purged walk-forward split — identical to train_xgb_alpha_v3.py."""
    test_mask = df['year'] == test_year
    train_mask = df['year'] < test_year

    train_dates = df.loc[train_mask, 'date']
    if len(train_dates) == 0:
        return None, None

    unique_train_dates = sorted(train_dates.unique())
    if len(unique_train_dates) <= PURGE_DAYS:
        return None, None

    purge_cutoff = unique_train_dates[-(PURGE_DAYS + 1)]
    train_mask_purged = (df['year'] < test_year) & (df['date'] <= purge_cutoff)

    return df.loc[train_mask_purged].copy(), df.loc[test_mask].copy()


def run_cv(df: pd.DataFrame, params: dict) -> tuple[float, float, list]:
    """
    Walk-forward CV with given hyperparameters.
    Returns (mean_ic, std_ic, yearly_ics).
    Objective is mean classification IC — same metric used for v2 tuning.
    """
    yearly_ics = []

    for test_year in CV_TEST_YEARS:
        train_df, test_df = get_train_test_split(df, test_year)

        if train_df is None or len(train_df) < 1000:
            continue

        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        y_train = train_df[TARGET_CLF].values
        y_test_reg = test_df[TARGET_REG].values

        # Fill missing with training median
        for col in FEATURES:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

        # Validation split for early stopping (last 10% of train)
        n_val = int(len(X_train) * 0.1)
        X_tr = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_tr = y_train[:-n_val]
        y_val = y_train[-n_val:]

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
            **FIXED_PARAMS,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict_proba(X_test)[:, 1]
        ic = calculate_ic(y_test_reg, y_pred)

        if not np.isnan(ic):
            yearly_ics.append(ic)

    if not yearly_ics:
        return 0.0, 0.0, []

    return float(np.mean(yearly_ics)), float(np.std(yearly_ics)), yearly_ics


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 2, 6),
        'subsample':        trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 500),
    }

    mean_ic, std_ic, yearly_ics = run_cv(df, params)

    trial.set_user_attr('std_ic', std_ic)
    trial.set_user_attr('yearly_ics', yearly_ics)

    return mean_ic


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost v3 (31 features) with Optuna"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--output-dir", default="scripts/ml/outputs",
        help="Output directory for results and model params",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous study from optuna_study_v3.db",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_path = output_dir / "optuna_study_v3.db"

    logger.info("=" * 70)
    logger.info("XGBOOST V3 HYPERPARAMETER TUNING (OPTUNA)")
    logger.info("=" * 70)
    logger.info(f"Database:    {args.db}")
    logger.info(f"Trials:      {args.n_trials}")
    logger.info(f"Features:    {len(FEATURES)} (v3 set)")
    logger.info(f"CV years:    {CV_TEST_YEARS}")
    logger.info(f"Output dir:  {output_dir}")

    # Load data once — shared across all trials
    df = load_data(args.db)

    # -----------------------------------------------------------------
    # Baseline: v2_tuned params on v3 features (what train_v3 uses now)
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE: v2_tuned params on 31 features")
    logger.info("=" * 70)
    baseline_ic, baseline_std, baseline_yearly = run_cv(df, V2_TUNED_PARAMS)
    logger.info(f"Mean IC: {baseline_ic:.4f}  Std: {baseline_std:.4f}")
    logger.info(f"Yearly:  {[f'{ic:.4f}' for ic in baseline_yearly]}")

    # -----------------------------------------------------------------
    # Create or resume study
    # -----------------------------------------------------------------
    storage = f"sqlite:///{study_path}"

    if args.resume and study_path.exists():
        logger.info(f"\nResuming study from {study_path}")
        study = optuna.load_study(study_name="xgb_v3_tuning", storage=storage)
        logger.info(f"Previous trials: {len(study.trials)}")
    else:
        logger.info("\nCreating new study")
        study = optuna.create_study(
            study_name="xgb_v3_tuning",
            direction="maximize",
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=False,
        )

    # -----------------------------------------------------------------
    # Per-trial progress callback
    # -----------------------------------------------------------------
    def callback(study, trial):
        if trial.value is not None:
            logger.info(
                f"Trial {trial.number:3d}: IC={trial.value:.4f} "
                f"(best={study.best_value:.4f})"
            )

    # -----------------------------------------------------------------
    # Run optimisation
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STARTING OPTIMISATION")
    logger.info("=" * 70)
    logger.info(
        f"Estimated time: {args.n_trials * 4}-{args.n_trials * 6} minutes "
        f"(31 features × 7 folds is slower than v2)"
    )

    start_time = datetime.now()

    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=args.n_trials,
        callbacks=[callback],
        show_progress_bar=True,
    )

    elapsed = datetime.now() - start_time

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMISATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time:        {elapsed}")
    logger.info(f"Trials completed:  {len(study.trials)}")

    logger.info("\n" + "-" * 40)
    logger.info("BEST PARAMETERS:")
    logger.info("-" * 40)
    for param, value in study.best_params.items():
        baseline_val = V2_TUNED_PARAMS.get(param, 'N/A')
        if isinstance(value, float):
            logger.info(f"  {param}: {value:.4f}  (v2_tuned was {baseline_val})")
        else:
            logger.info(f"  {param}: {value}  (v2_tuned was {baseline_val})")

    logger.info("\n" + "-" * 40)
    logger.info("PERFORMANCE vs BASELINE:")
    logger.info("-" * 40)
    logger.info(f"  Baseline IC (v2_tuned params): {baseline_ic:.4f}")
    logger.info(f"  Best IC    (v3_tuned params):  {study.best_value:.4f}")
    improvement = (study.best_value - baseline_ic) / max(abs(baseline_ic), 1e-9) * 100
    logger.info(f"  Improvement: {improvement:+.1f}%")

    # -----------------------------------------------------------------
    # Save best params — ready to drop into train_xgb_alpha_v3.py
    # -----------------------------------------------------------------
    best_params_full = {**study.best_params, **FIXED_PARAMS}
    best_params_path = output_dir / 'best_params_v3_tuned.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_params_full, f, indent=2)
    logger.info(f"\nSaved best params: {best_params_path}")

    # -----------------------------------------------------------------
    # Save all trials to CSV
    # -----------------------------------------------------------------
    trials_data = []
    for trial in study.trials:
        if trial.value is not None:
            row = {
                'trial': trial.number,
                'mean_ic': trial.value,
                'std_ic': trial.user_attrs.get('std_ic', np.nan),
                **trial.params,
            }
            trials_data.append(row)

    trials_df = pd.DataFrame(trials_data).sort_values('mean_ic', ascending=False)
    trials_path = output_dir / 'tuning_results_v3.csv'
    trials_df.to_csv(trials_path, index=False)
    logger.info(f"Saved all trials:  {trials_path}")

    # -----------------------------------------------------------------
    # Top 5
    # -----------------------------------------------------------------
    logger.info("\n" + "-" * 40)
    logger.info("TOP 5 TRIALS:")
    logger.info("-" * 40)
    print(trials_df.head().to_string(index=False))

    # -----------------------------------------------------------------
    # Next steps
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info(
        f"Update XGB_PARAMS_CLF and XGB_PARAMS_REG in "
        f"train_xgb_alpha_v3.py with the values from:"
    )
    logger.info(f"  {best_params_path}")
    logger.info("")
    logger.info("Then retrain:")
    logger.info(
        f"  python scripts/ml/train_xgb_alpha_v3.py "
        f"--db {args.db} --skip-shap"
    )
    logger.info("")
    logger.info("To run more trials:")
    logger.info(
        f"  python scripts/ml/tune_xgb_optuna_v3.py "
        f"--db {args.db} --n-trials 50 --resume"
    )


if __name__ == "__main__":
    main()