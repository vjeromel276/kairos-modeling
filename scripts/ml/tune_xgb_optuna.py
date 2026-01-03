#!/usr/bin/env python3
"""
tune_xgb_optuna.py
==================
Hyperparameter tuning for XGBoost ML v2 using Optuna.

Uses the same 23 features as v2, same walk-forward CV structure,
but searches for optimal hyperparameters.

Search space:
- learning_rate: [0.005 - 0.1] (log scale)
- max_depth: [2 - 6]
- subsample: [0.5 - 0.9]
- colsample_bytree: [0.5 - 0.9]
- reg_lambda: [0.1 - 10] (log scale)
- reg_alpha: [0 - 1]
- min_child_weight: [50 - 500]

Objective: Maximize mean classification IC across CV years (2019-2024)

Usage:
    # Run 50 trials (~3-4 hours)
    python scripts/ml/tune_xgb_optuna.py --db data/kairos.duckdb --n-trials 50
    
    # Quick test with 5 trials
    python scripts/ml/tune_xgb_optuna.py --db data/kairos.duckdb --n-trials 5
    
    # Resume previous study
    python scripts/ml/tune_xgb_optuna.py --db data/kairos.duckdb --n-trials 50 --resume

Output:
    - best_params_v2_tuned.json: Best hyperparameters found
    - optuna_study.db: SQLite database with all trial results (for resume)
    - tuning_results.csv: All trials with their scores
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

# Suppress XGBoost warnings during tuning
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# FEATURES (same as v2 - 23 features)
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

TARGET_REG = 'ret_5d_f'
TARGET_CLF = 'label_5d_up'

# Walk-forward CV settings
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PURGE_DAYS = 5

# Fixed params (not tuned)
FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50


def load_data(db_path: str) -> pd.DataFrame:
    """Load features from database."""
    
    logger.info("Loading data...")
    con = duckdb.connect(db_path, read_only=True)
    
    # Base grid
    df = con.execute("""
        SELECT ticker, date, ret_5d_f
        FROM feat_targets
        WHERE date >= '2015-01-01'
          AND ret_5d_f IS NOT NULL
    """).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    
    # Join feature tables
    for table, cols in FEATURE_SOURCES.items():
        cols_str = ', '.join(cols)
        feat_df = con.execute(f"""
            SELECT ticker, date, {cols_str}
            FROM {table}
        """).fetchdf()
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        df = df.merge(feat_df, on=['ticker', 'date'], how='left')
    
    con.close()
    
    # Create targets
    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    df['year'] = df['date'].dt.year
    
    logger.info(f"Loaded {len(df):,} rows, {len(FEATURES)} features")
    
    return df


def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman IC."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def get_train_test_split(df: pd.DataFrame, test_year: int):
    """Split with purging."""
    
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


def run_cv_with_params(df: pd.DataFrame, params: dict) -> tuple:
    """
    Run walk-forward CV with given hyperparameters.
    Returns (mean_ic, std_ic, yearly_ics)
    """
    
    yearly_ics = []
    
    for test_year in CV_TEST_YEARS:
        train_df, test_df = get_train_test_split(df, test_year)
        
        if train_df is None or len(train_df) < 1000:
            continue
        
        # Prepare features
        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        
        y_train = train_df[TARGET_CLF].values
        y_test_reg = test_df[TARGET_REG].values  # For IC calculation
        y_test_clf = test_df[TARGET_CLF].values
        
        # Fill missing with median
        for col in FEATURES:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
        
        # Validation split for early stopping
        n_val = int(len(X_train) * 0.1)
        X_train_fit = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_train_fit = y_train[:-n_val]
        y_val = y_train[-n_val:]
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
            **FIXED_PARAMS
        )
        
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict and calculate IC
        y_pred = model.predict_proba(X_test)[:, 1]
        ic = calculate_ic(y_test_reg, y_pred)
        
        if not np.isnan(ic):
            yearly_ics.append(ic)
    
    if len(yearly_ics) == 0:
        return 0.0, 0.0, []
    
    return np.mean(yearly_ics), np.std(yearly_ics), yearly_ics


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """Optuna objective function."""
    
    # Sample hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 500),
    }
    
    # Run CV
    mean_ic, std_ic, yearly_ics = run_cv_with_params(df, params)
    
    # Store additional info
    trial.set_user_attr('std_ic', std_ic)
    trial.set_user_attr('yearly_ics', yearly_ics)
    
    return mean_ic


def main():
    parser = argparse.ArgumentParser(description="Tune XGBoost v2 with Optuna")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--output-dir", default="scripts/ml/outputs", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume previous study")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    study_path = output_dir / "optuna_study.db"
    
    logger.info("=" * 70)
    logger.info("XGBOOST HYPERPARAMETER TUNING (OPTUNA)")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Features: {len(FEATURES)} (v2 feature set)")
    logger.info(f"CV Years: {CV_TEST_YEARS}")
    
    # Load data once
    df = load_data(args.db)
    
    # Create or load study
    storage = f"sqlite:///{study_path}"
    
    if args.resume and study_path.exists():
        logger.info(f"Resuming study from {study_path}")
        study = optuna.load_study(study_name="xgb_tuning", storage=storage)
        logger.info(f"Previous trials: {len(study.trials)}")
    else:
        logger.info("Creating new study")
        study = optuna.create_study(
            study_name="xgb_tuning",
            direction="maximize",
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=False
        )
    
    # Current best (v2 defaults) for comparison
    v2_defaults = {
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'min_child_weight': 100,
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE: v2 Default Parameters")
    logger.info("=" * 70)
    baseline_ic, baseline_std, baseline_yearly = run_cv_with_params(df, v2_defaults)
    logger.info(f"Mean IC: {baseline_ic:.4f} (+/- {baseline_std:.4f})")
    logger.info(f"Yearly:  {[f'{ic:.4f}' for ic in baseline_yearly]}")
    
    # Progress callback
    def callback(study, trial):
        if trial.value is not None:
            logger.info(
                f"Trial {trial.number}: IC={trial.value:.4f} "
                f"(best={study.best_value:.4f})"
            )
    
    # Run optimization
    logger.info("\n" + "=" * 70)
    logger.info("STARTING OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Estimated time: {args.n_trials * 3}-{args.n_trials * 4} minutes")
    
    start_time = datetime.now()
    
    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=args.n_trials,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    elapsed = datetime.now() - start_time
    
    # Results
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Trials completed: {len(study.trials)}")
    
    logger.info("\n" + "-" * 40)
    logger.info("BEST PARAMETERS:")
    logger.info("-" * 40)
    for param, value in study.best_params.items():
        default = v2_defaults.get(param, 'N/A')
        logger.info(f"  {param}: {value:.4f} (was {default})")
    
    logger.info("\n" + "-" * 40)
    logger.info("PERFORMANCE COMPARISON:")
    logger.info("-" * 40)
    logger.info(f"  Baseline IC: {baseline_ic:.4f}")
    logger.info(f"  Best IC:     {study.best_value:.4f}")
    improvement = (study.best_value - baseline_ic) / baseline_ic * 100
    logger.info(f"  Improvement: {improvement:+.1f}%")
    
    # Save best params
    best_params_full = {**study.best_params, **FIXED_PARAMS}
    best_params_path = output_dir / 'best_params_v2_tuned.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_params_full, f, indent=2)
    logger.info(f"\nSaved best params to: {best_params_path}")
    
    # Save all trials to CSV
    trials_data = []
    for trial in study.trials:
        if trial.value is not None:
            row = {
                'trial': trial.number,
                'mean_ic': trial.value,
                'std_ic': trial.user_attrs.get('std_ic', np.nan),
                **trial.params
            }
            trials_data.append(row)
    
    trials_df = pd.DataFrame(trials_data)
    trials_df = trials_df.sort_values('mean_ic', ascending=False)
    trials_path = output_dir / 'tuning_results.csv'
    trials_df.to_csv(trials_path, index=False)
    logger.info(f"Saved all trials to: {trials_path}")
    
    # Show top 5 trials
    logger.info("\n" + "-" * 40)
    logger.info("TOP 5 TRIALS:")
    logger.info("-" * 40)
    print(trials_df.head().to_string(index=False))
    
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("To train final model with tuned params, update train_xgb_alpha_v2.py")
    logger.info(f"with parameters from: {best_params_path}")
    logger.info("")
    logger.info("Or run more trials to continue searching:")
    logger.info(f"  python {__file__} --db {args.db} --n-trials 50 --resume")


if __name__ == "__main__":
    main()