#!/usr/bin/env python3
"""
train_xgb_alpha_v2_tuned.py
===========================
XGBoost ML v2 with Optuna-tuned hyperparameters.

Changes from v2:
- Hyperparameters optimized via 105 Optuna trials
- Mean IC improved from 0.0352 to 0.0415 (+17.9%)
- Std IC reduced from 0.0158 to 0.0087 (more consistent)

Key parameter changes:
- learning_rate: 0.03 -> 0.10 (3x faster)
- subsample: 0.7 -> 0.55 (more aggressive sampling)
- colsample_bytree: 0.7 -> 0.86 (use more features)
- reg_alpha: 0.1 -> 0.64 (6x more L1 regularization)
- min_child_weight: 100 -> 408 (4x stricter leaf requirements)

Usage:
    python scripts/ml/train_xgb_alpha_v2_tuned.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_alpha_v2_tuned.py --db data/kairos.duckdb --skip-cv
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Feature importance plots will be skipped.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# RAW FEATURES (23 features - same as v2)
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

# Walk-forward CV
CV_START_YEAR = 2015
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PURGE_DAYS = 5

# =============================================================================
# TUNED HYPERPARAMETERS (from Optuna Trial 78)
# =============================================================================

XGB_PARAMS_REG = {
    'learning_rate': 0.0996,       # was 0.03
    'max_depth': 4,                # unchanged
    'subsample': 0.545,            # was 0.7
    'colsample_bytree': 0.857,     # was 0.7
    'reg_lambda': 0.50,            # was 1.0
    'reg_alpha': 0.644,            # was 0.1
    'min_child_weight': 408,       # was 100
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

XGB_PARAMS_CLF = {
    'learning_rate': 0.0996,       # was 0.03
    'max_depth': 4,                # unchanged
    'subsample': 0.545,            # was 0.7
    'colsample_bytree': 0.857,     # was 0.7
    'reg_lambda': 0.50,            # was 1.0
    'reg_alpha': 0.644,            # was 0.1
    'min_child_weight': 408,       # was 100
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50


def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load raw features from multiple tables and join."""
    
    logger.info("Loading raw features from multiple tables...")
    
    base_query = """
        SELECT ticker, date, ret_5d_f
        FROM feat_targets
        WHERE date >= '2015-01-01'
          AND ret_5d_f IS NOT NULL
    """
    df = con.execute(base_query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Base grid: {len(df):,} rows")
    
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
    
    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    df['year'] = df['date'].dt.year
    
    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(FEATURES)} features")
    
    return df


def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Information Coefficient (Spearman correlation)."""
    from scipy.stats import spearmanr
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def get_train_test_split(df: pd.DataFrame, test_year: int, purge_days: int = 5):
    """Split data for walk-forward CV with purging."""
    
    test_mask = df['year'] == test_year
    train_mask = df['year'] < test_year
    
    train_dates = df.loc[train_mask, 'date']
    if len(train_dates) == 0:
        return None, None
    
    unique_train_dates = sorted(train_dates.unique())
    if len(unique_train_dates) <= purge_days:
        return None, None
    
    purge_cutoff = unique_train_dates[-(purge_days + 1)]
    train_mask_purged = (df['year'] < test_year) & (df['date'] <= purge_cutoff)
    
    train_df = df.loc[train_mask_purged].copy()
    test_df = df.loc[test_mask].copy()
    
    return train_df, test_df


def run_walk_forward_cv(df: pd.DataFrame, output_dir: Path):
    """Run purged walk-forward cross-validation."""
    
    results = []
    
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD CV (v2 TUNED)")
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
        X_test = test_df[FEATURES].copy()
        
        y_train_reg = train_df[TARGET_REG].values
        y_test_reg = test_df[TARGET_REG].values
        
        y_train_clf = train_df[TARGET_CLF].values
        y_test_clf = test_df[TARGET_CLF].values
        
        medians = {}
        for col in FEATURES:
            medians[col] = X_train[col].median()
            X_train[col] = X_train[col].fillna(medians[col])
            X_test[col] = X_test[col].fillna(medians[col])
        
        n_train = len(X_train)
        n_val = int(n_train * 0.1)
        
        X_train_fit = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_train_fit_reg = y_train_reg[:-n_val]
        y_val_reg = y_train_reg[-n_val:]
        y_train_fit_clf = y_train_clf[:-n_val]
        y_val_clf = y_train_clf[-n_val:]
        
        # Regression
        logger.info("Training regression model...")
        
        model_reg = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_REG
        )
        
        model_reg.fit(
            X_train_fit, y_train_fit_reg,
            eval_set=[(X_val, y_val_reg)],
            verbose=False
        )
        
        y_pred_reg = model_reg.predict(X_test)
        ic_reg = calculate_ic(y_test_reg, y_pred_reg)
        logger.info(f"Regression IC: {ic_reg:.4f} (trees: {model_reg.best_iteration})")
        
        # Classification
        logger.info("Training classification model...")
        
        model_clf = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_CLF
        )
        
        model_clf.fit(
            X_train_fit, y_train_fit_clf,
            eval_set=[(X_val, y_val_clf)],
            verbose=False
        )
        
        y_pred_clf = model_clf.predict_proba(X_test)[:, 1]
        ic_clf = calculate_ic(y_test_reg, y_pred_clf)
        auc_clf = roc_auc_score(y_test_clf, y_pred_clf)
        
        logger.info(f"Classification IC: {ic_clf:.4f}, AUC: {auc_clf:.4f}")
        
        results.append({
            'test_year': test_year,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'ic_regression': ic_reg,
            'ic_classification': ic_clf,
            'auc_classification': auc_clf,
            'n_trees_reg': model_reg.best_iteration,
            'n_trees_clf': model_clf.best_iteration,
        })
    
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results_v2_tuned.csv'
    results_df.to_csv(results_path, index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("CV SUMMARY (v2 TUNED)")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    logger.info(f"\nMean IC (Regression):     {results_df['ic_regression'].mean():.4f}")
    logger.info(f"Mean IC (Classification): {results_df['ic_classification'].mean():.4f}")
    logger.info(f"Std IC (Classification):  {results_df['ic_classification'].std():.4f}")
    
    return results_df


def train_final_models(df: pd.DataFrame, output_dir: Path):
    """Train final models on all data."""
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL MODELS ON ALL DATA (TUNED)")
    logger.info("=" * 70)
    
    X = df[FEATURES].copy()
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values
    
    medians = {}
    for col in FEATURES:
        medians[col] = X[col].median()
        X[col] = X[col].fillna(medians[col])
    
    medians_path = output_dir / 'feature_medians_v2_tuned.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    
    n = len(X)
    n_val = int(n * 0.1)
    
    X_train = X.iloc[:-n_val]
    X_val = X.iloc[-n_val:]
    
    # Regression
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
    
    reg_path = output_dir / 'model_regression_v2_tuned.json'
    model_reg.save_model(str(reg_path))
    logger.info(f"Saved: {reg_path} (trees: {model_reg.best_iteration})")
    
    # Classification
    logger.info(f"Training final classification model...")
    
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
    
    clf_path = output_dir / 'model_classification_v2_tuned.json'
    model_clf.save_model(str(clf_path))
    logger.info(f"Saved: {clf_path} (trees: {model_clf.best_iteration})")
    
    return model_reg, model_clf, medians


def generate_shap_analysis(model_reg, model_clf, df: pd.DataFrame, output_dir: Path):
    """Generate SHAP feature importance."""
    
    if not HAS_SHAP:
        logger.warning("SHAP not installed, skipping")
        return
    
    logger.info("\nGenerating SHAP analysis...")
    
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample = sample_df[FEATURES].copy()
    for col in FEATURES:
        X_sample[col] = X_sample[col].fillna(X_sample[col].median())
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        explainer_reg = shap.TreeExplainer(model_reg)
        shap_values_reg = explainer_reg.shap_values(X_sample)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_reg, X_sample, show=False)
        plt.title("SHAP Feature Importance - Regression (v2 Tuned)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression_v2_tuned.png', dpi=150)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_reg, X_sample, plot_type="bar", show=False)
        plt.title("Mean |SHAP| - Regression (v2 Tuned)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression_bar_v2_tuned.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved SHAP plots to {output_dir}")
        
    except Exception as e:
        logger.error(f"SHAP error: {e}")


def generate_predictions(con, model_reg, model_clf, medians: dict, output_dir: Path):
    """Generate predictions and save to DuckDB."""
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING PREDICTIONS (v2 TUNED)")
    logger.info("=" * 70)
    
    df = load_data(con)
    
    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = X[col].fillna(medians.get(col, 0))
    
    logger.info("Generating predictions...")
    df['alpha_ml_v2_tuned_reg'] = model_reg.predict(X)
    df['alpha_ml_v2_tuned_clf'] = model_clf.predict_proba(X)[:, 1]
    
    output_df = df[['ticker', 'date', 'alpha_ml_v2_tuned_reg', 'alpha_ml_v2_tuned_clf']]
    
    logger.info("Saving to feat_alpha_ml_xgb_v2_tuned...")
    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb_v2_tuned")
    con.register("predictions_df", output_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb_v2_tuned AS 
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)
    
    count = con.execute("SELECT COUNT(*) FROM feat_alpha_ml_xgb_v2_tuned").fetchone()[0]
    logger.info(f"Created feat_alpha_ml_xgb_v2_tuned with {count:,} rows")
    
    output_df.to_parquet(output_dir / 'predictions_v2_tuned.parquet', index=False)
    
    return output_df


def validate_vs_baseline(con):
    """Compare to baseline and original v2."""
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION: v2 TUNED vs v2 vs BASELINE")
    logger.info("=" * 70)
    
    # Check which tables exist
    v2_exists = con.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'feat_alpha_ml_xgb_v2'
    """).fetchone()[0] > 0
    
    if v2_exists:
        comparison = con.execute("""
            SELECT 
                CORR(m.alpha_composite_v8, t.ret_5d_f) as ic_baseline,
                CORR(v2.alpha_ml_v2_clf, t.ret_5d_f) as ic_v2,
                CORR(tuned.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_tuned,
                COUNT(*) as n
            FROM feat_matrix_v2 m
            JOIN feat_alpha_ml_xgb_v2 v2 ON m.ticker = v2.ticker AND m.date = v2.date
            JOIN feat_alpha_ml_xgb_v2_tuned tuned ON m.ticker = tuned.ticker AND m.date = tuned.date
            JOIN feat_targets t ON m.ticker = t.ticker AND m.date = t.date
            WHERE m.date >= '2015-01-01'
              AND t.ret_5d_f IS NOT NULL
              AND m.alpha_composite_v8 IS NOT NULL
        """).fetchone()
        
        logger.info(f"\nOverall IC:")
        logger.info(f"  Baseline v8:   {comparison[0]:.4f}")
        logger.info(f"  ML v2:         {comparison[1]:.4f}")
        logger.info(f"  ML v2 Tuned:   {comparison[2]:.4f}")
        logger.info(f"  (n = {comparison[3]:,})")
        
        improvement = (comparison[2] - comparison[1]) / abs(comparison[1]) * 100
        logger.info(f"\n  Tuned vs v2 improvement: {improvement:+.1f}%")
        
        # By year
        logger.info("\nIC by Year:")
        yearly = con.execute("""
            SELECT 
                YEAR(m.date) as year,
                CORR(m.alpha_composite_v8, t.ret_5d_f) as ic_v8,
                CORR(v2.alpha_ml_v2_clf, t.ret_5d_f) as ic_v2,
                CORR(tuned.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_tuned,
                COUNT(*) as n
            FROM feat_matrix_v2 m
            JOIN feat_alpha_ml_xgb_v2 v2 ON m.ticker = v2.ticker AND m.date = v2.date
            JOIN feat_alpha_ml_xgb_v2_tuned tuned ON m.ticker = tuned.ticker AND m.date = tuned.date
            JOIN feat_targets t ON m.ticker = t.ticker AND m.date = t.date
            WHERE m.date >= '2015-01-01'
              AND t.ret_5d_f IS NOT NULL
              AND m.alpha_composite_v8 IS NOT NULL
            GROUP BY YEAR(m.date)
            ORDER BY year
        """).fetchdf()
        
        logger.info(f"\n{'Year':<6} {'v8':>10} {'v2':>10} {'Tuned':>10}")
        logger.info("-" * 40)
        for _, row in yearly.iterrows():
            logger.info(f"{int(row['year']):<6} {row['ic_v8']:>+10.4f} {row['ic_v2']:>+10.4f} {row['ic_tuned']:>+10.4f}")
    
    else:
        comparison = con.execute("""
            SELECT 
                CORR(m.alpha_composite_v8, t.ret_5d_f) as ic_baseline,
                CORR(tuned.alpha_ml_v2_tuned_reg, t.ret_5d_f) as ic_tuned_reg,
                CORR(tuned.alpha_ml_v2_tuned_clf, t.ret_5d_f) as ic_tuned_clf,
                COUNT(*) as n
            FROM feat_matrix_v2 m
            JOIN feat_alpha_ml_xgb_v2_tuned tuned ON m.ticker = tuned.ticker AND m.date = tuned.date
            JOIN feat_targets t ON m.ticker = t.ticker AND m.date = t.date
            WHERE m.date >= '2015-01-01'
              AND t.ret_5d_f IS NOT NULL
              AND m.alpha_composite_v8 IS NOT NULL
        """).fetchone()
        
        logger.info(f"\nOverall IC:")
        logger.info(f"  Baseline v8:          {comparison[0]:.4f}")
        logger.info(f"  ML v2 Tuned Reg:      {comparison[1]:.4f}")
        logger.info(f"  ML v2 Tuned Clf:      {comparison[2]:.4f}")
        logger.info(f"  (n = {comparison[3]:,})")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost v2 with tuned hyperparameters")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="scripts/ml/outputs", help="Output directory")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV, only train final")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("XGBOOST ALPHA v2 TUNED (OPTUNA OPTIMIZED)")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Features: {len(FEATURES)}")
    logger.info("")
    logger.info("Tuned hyperparameters (vs defaults):")
    logger.info(f"  learning_rate:    0.0996 (was 0.03)")
    logger.info(f"  max_depth:        4 (unchanged)")
    logger.info(f"  subsample:        0.545 (was 0.7)")
    logger.info(f"  colsample_bytree: 0.857 (was 0.7)")
    logger.info(f"  reg_lambda:       0.50 (was 1.0)")
    logger.info(f"  reg_alpha:        0.644 (was 0.1)")
    logger.info(f"  min_child_weight: 408 (was 100)")
    
    con = duckdb.connect(args.db)
    
    try:
        df = load_data(con)
        
        if not args.skip_cv:
            cv_results = run_walk_forward_cv(df, output_dir)
        
        model_reg, model_clf, medians = train_final_models(df, output_dir)
        
        if not args.skip_shap and HAS_SHAP:
            generate_shap_analysis(model_reg, model_clf, df, output_dir)
        
        generate_predictions(con, model_reg, model_clf, medians, output_dir)
        
        validate_vs_baseline(con)
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Outputs: {output_dir}")
        logger.info("  - cv_results_v2_tuned.csv")
        logger.info("  - model_regression_v2_tuned.json")
        logger.info("  - model_classification_v2_tuned.json")
        logger.info("  - feature_medians_v2_tuned.json")
        logger.info("  - predictions_v2_tuned.parquet")
        logger.info("DuckDB table: feat_alpha_ml_xgb_v2_tuned")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()