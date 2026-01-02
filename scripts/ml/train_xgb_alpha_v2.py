#!/usr/bin/env python3
"""
train_xgb_alpha_v2.py
=====================
XGBoost ML v2 - Train on RAW features instead of pre-blended composites.

Key difference from v1:
- v1 used composites (value_composite_z, quality_composite_z) with baked-in sign assumptions
- v2 uses raw features and lets ML learn the optimal combinations and signs

Features (23 total, all >90% coverage, |IC|>0.005):
- Fundamental: earnings_yield, fcf_yield, roa, book_to_market, operating_margin, roe
- Volatility: vol_21, vol_63, vol_blend, resid_vol_63d
- Beta: beta_21d, beta_63d, beta_252d
- Price Action: hl_ratio, range_pct, ret_21d, ret_5d
- Momentum: mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1, reversal_1m

Usage:
    python scripts/ml/train_xgb_alpha_v2.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_alpha_v2.py --db data/kairos.duckdb --skip-shap
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
# RAW FEATURES (23 features with >90% coverage and |IC|>0.005)
# =============================================================================

FEATURE_SOURCES = {
    # Table: [columns]
    'feat_fundamental': [
        'earnings_yield',    # IC: -0.035 (strong, inverted)
        'fcf_yield',         # IC: -0.018
        'roa',               # IC: -0.015
        'book_to_market',    # IC: +0.011
        'operating_margin',  # IC: -0.006
        'roe',               # IC: -0.005
    ],
    'feat_vol_sizing': [
        'vol_21',            # IC: +0.030
        'vol_63',            # IC: +0.023
        'vol_blend',         # IC: +0.029
    ],
    'feat_beta': [
        'beta_21d',          # IC: +0.023
        'beta_63d',          # IC: +0.010
        'beta_252d',         # IC: +0.016
        'resid_vol_63d',     # IC: +0.024
    ],
    'feat_price_action': [
        'hl_ratio',          # IC: +0.027
        'range_pct',         # IC: +0.022
        'ret_21d',           # IC: +0.018
        'ret_5d',            # IC: -0.009
    ],
    'feat_momentum_v2': [
        'mom_1m',            # IC: -0.012
        'mom_3m',            # IC: -0.013
        'mom_6m',            # IC: -0.009
        'mom_12m',           # IC: -0.011
        'mom_12_1',          # IC: -0.007
        'reversal_1m',       # IC: +0.012
    ],
}

# Flatten to list
FEATURES = []
for cols in FEATURE_SOURCES.values():
    FEATURES.extend(cols)

TARGET_REG = 'ret_5d_f'
TARGET_CLF = 'label_5d_up'

# Walk-forward CV
CV_START_YEAR = 2015
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PURGE_DAYS = 5

# XGBoost params (tuned for financial data)
XGB_PARAMS_REG = {
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'min_child_weight': 100,  # Require more samples per leaf
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

XGB_PARAMS_CLF = {
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'min_child_weight': 100,
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
    
    # Start with base grid from feat_targets (has ticker, date, ret_5d_f)
    base_query = """
        SELECT ticker, date, ret_5d_f
        FROM feat_targets
        WHERE date >= '2015-01-01'
          AND ret_5d_f IS NOT NULL
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
        
        # Log coverage
        for col in cols:
            cov = df[col].notna().mean() * 100
            logger.info(f"    {col}: {cov:.1f}%")
    
    # Create classification target
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
    logger.info("WALK-FORWARD CV (RAW FEATURES)")
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
        
        # Prepare features
        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        
        y_train_reg = train_df[TARGET_REG].values
        y_test_reg = test_df[TARGET_REG].values
        
        y_train_clf = train_df[TARGET_CLF].values
        y_test_clf = test_df[TARGET_CLF].values
        
        # Fill missing with median from training
        medians = {}
        for col in FEATURES:
            medians[col] = X_train[col].median()
            X_train[col] = X_train[col].fillna(medians[col])
            X_test[col] = X_test[col].fillna(medians[col])
        
        # Validation split for early stopping
        n_train = len(X_train)
        n_val = int(n_train * 0.1)
        
        X_train_fit = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_train_fit_reg = y_train_reg[:-n_val]
        y_val_reg = y_train_reg[-n_val:]
        y_train_fit_clf = y_train_clf[:-n_val]
        y_val_clf = y_train_clf[-n_val:]
        
        # =================================================================
        # REGRESSION MODEL
        # =================================================================
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
        
        # =================================================================
        # CLASSIFICATION MODEL
        # =================================================================
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
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results_v2.csv'
    results_df.to_csv(results_path, index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("CV SUMMARY (RAW FEATURES)")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    logger.info(f"\nMean IC (Regression):     {results_df['ic_regression'].mean():.4f}")
    logger.info(f"Mean IC (Classification): {results_df['ic_classification'].mean():.4f}")
    
    return results_df


def train_final_models(df: pd.DataFrame, output_dir: Path):
    """Train final models on all data."""
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL MODELS ON ALL DATA")
    logger.info("=" * 70)
    
    X = df[FEATURES].copy()
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values
    
    # Fill and save medians
    medians = {}
    for col in FEATURES:
        medians[col] = X[col].median()
        X[col] = X[col].fillna(medians[col])
    
    medians_path = output_dir / 'feature_medians_v2.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    
    # Split for early stopping
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
    
    reg_path = output_dir / 'model_regression_v2.json'
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
    
    clf_path = output_dir / 'model_classification_v2.json'
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
        
        # Regression SHAP
        explainer_reg = shap.TreeExplainer(model_reg)
        shap_values_reg = explainer_reg.shap_values(X_sample)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_reg, X_sample, show=False)
        plt.title("SHAP Feature Importance - Regression (Raw Features)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression_v2.png', dpi=150)
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_reg, X_sample, plot_type="bar", show=False)
        plt.title("Mean |SHAP| - Regression")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression_bar_v2.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved SHAP plots to {output_dir}")
        
    except Exception as e:
        logger.error(f"SHAP error: {e}")


def generate_predictions(con, model_reg, model_clf, medians: dict, output_dir: Path):
    """Generate predictions and save to DuckDB."""
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 70)
    
    # Load all data
    df = load_data(con)
    
    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = X[col].fillna(medians.get(col, 0))
    
    logger.info("Generating predictions...")
    df['alpha_ml_v2_reg'] = model_reg.predict(X)
    df['alpha_ml_v2_clf'] = model_clf.predict_proba(X)[:, 1]
    
    output_df = df[['ticker', 'date', 'alpha_ml_v2_reg', 'alpha_ml_v2_clf']]
    
    # Save to DuckDB
    logger.info("Saving to feat_alpha_ml_xgb_v2...")
    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb_v2")
    con.register("predictions_df", output_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb_v2 AS 
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)
    
    count = con.execute("SELECT COUNT(*) FROM feat_alpha_ml_xgb_v2").fetchone()[0]
    logger.info(f"Created feat_alpha_ml_xgb_v2 with {count:,} rows")
    
    # Save parquet backup
    output_df.to_parquet(output_dir / 'predictions_v2.parquet', index=False)
    
    return output_df


def validate_vs_baseline(con):
    """Compare to baseline alpha_composite_v8."""
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION: ML v2 vs BASELINE")
    logger.info("=" * 70)
    
    comparison = con.execute("""
        SELECT 
            CORR(m.alpha_composite_v8, t.ret_5d_f) as ic_baseline,
            CORR(ml.alpha_ml_v2_reg, t.ret_5d_f) as ic_ml_reg,
            CORR(ml.alpha_ml_v2_clf, t.ret_5d_f) as ic_ml_clf,
            COUNT(*) as n
        FROM feat_matrix_v2 m
        JOIN feat_alpha_ml_xgb_v2 ml ON m.ticker = ml.ticker AND m.date = ml.date
        JOIN feat_targets t ON m.ticker = t.ticker AND m.date = t.date
        WHERE m.date >= '2015-01-01'
          AND t.ret_5d_f IS NOT NULL
          AND m.alpha_composite_v8 IS NOT NULL
    """).fetchone()
    
    logger.info(f"\nOverall IC:")
    logger.info(f"  Baseline v8:      {comparison[0]:.4f}")
    logger.info(f"  ML v2 Regression: {comparison[1]:.4f}")
    logger.info(f"  ML v2 Classification: {comparison[2]:.4f}")
    logger.info(f"  (n = {comparison[3]:,})")
    
    # By year
    logger.info("\nIC by Year:")
    yearly = con.execute("""
        SELECT 
            YEAR(m.date) as year,
            CORR(m.alpha_composite_v8, t.ret_5d_f) as ic_v8,
            CORR(ml.alpha_ml_v2_reg, t.ret_5d_f) as ic_reg,
            CORR(ml.alpha_ml_v2_clf, t.ret_5d_f) as ic_clf,
            COUNT(*) as n
        FROM feat_matrix_v2 m
        JOIN feat_alpha_ml_xgb_v2 ml ON m.ticker = ml.ticker AND m.date = ml.date
        JOIN feat_targets t ON m.ticker = t.ticker AND m.date = t.date
        WHERE m.date >= '2015-01-01'
          AND t.ret_5d_f IS NOT NULL
          AND m.alpha_composite_v8 IS NOT NULL
        GROUP BY YEAR(m.date)
        ORDER BY year
    """).fetchdf()
    
    logger.info(f"\n{'Year':<6} {'v8':>10} {'ML_Reg':>10} {'ML_Clf':>10}")
    logger.info("-" * 40)
    for _, row in yearly.iterrows():
        logger.info(f"{int(row['year']):<6} {row['ic_v8']:>+10.4f} {row['ic_reg']:>+10.4f} {row['ic_clf']:>+10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost v2 on raw features")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="scripts/ml/outputs", help="Output directory")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV, only train final")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("XGBOOST ALPHA v2 (RAW FEATURES)")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Features: {len(FEATURES)}")
    logger.info(f"Feature list: {FEATURES}")
    
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
        logger.info("  - cv_results_v2.csv")
        logger.info("  - model_regression_v2.json")
        logger.info("  - model_classification_v2.json")
        logger.info("  - feature_medians_v2.json")
        logger.info("  - predictions_v2.parquet")
        logger.info("DuckDB table: feat_alpha_ml_xgb_v2")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()