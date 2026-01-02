#!/usr/bin/env python3
"""
train_xgb_alpha.py
==================
Train XGBoost models to combine alpha factors using nonlinear interactions.

Replaces linear IC-weighted blending with ML-based factor combination.

Models:
    1. Regression: Predict ret_5d_f directly
    2. Classification: Predict probability of positive return

Validation:
    Purged walk-forward CV with expanding window
    - Train: 2015 to year Y-1
    - Gap: 5 trading days (purge for ret_5d_f horizon)
    - Test: Year Y

Output:
    - feat_alpha_ml_xgb table in DuckDB
    - Model files (JSON)
    - SHAP importance plots
    - Validation metrics CSV

Usage:
    python scripts/ml/train_xgb_alpha.py --db data/kairos.duckdb
    python scripts/ml/train_xgb_alpha.py --db data/kairos.duckdb --skip-shap  # Faster
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Feature importance plots will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES = [
    'quality_composite_z',
    'value_composite_z', 
    'momentum_composite_z',
    'mom_12_1',
    'reversal_1m',
    'insider_composite_z',
    'vol_blend',
    'beta_252d',
    'adv_z',
]

TARGET_REG = 'ret_5d_f'
TARGET_CLF = 'label_5d_up'

# Walk-forward CV configuration
CV_START_YEAR = 2015
CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PURGE_DAYS = 5  # Gap between train and test to avoid lookahead

# XGBoost parameters (tuned for low SNR financial data)
XGB_PARAMS_REG = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

XGB_PARAMS_CLF = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load features and targets from feat_matrix_v2."""
    
    cols = FEATURES + [TARGET_REG, 'ticker', 'date']
    cols_str = ', '.join(cols)
    
    logger.info("Loading data from feat_matrix_v2...")
    
    df = con.execute(f"""
        SELECT {cols_str}
        FROM feat_matrix_v2
        WHERE date >= '2015-01-01'
          AND {TARGET_REG} IS NOT NULL
    """).fetchdf()
    
    # Create classification target
    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Years: {sorted(df['year'].unique())}")
    
    # Check feature coverage
    logger.info("\nFeature coverage:")
    for feat in FEATURES:
        coverage = df[feat].notna().mean() * 100
        logger.info(f"  {feat}: {coverage:.1f}%")
    
    return df


# =============================================================================
# WALK-FORWARD CV WITH PURGING
# =============================================================================

def get_train_test_split(df: pd.DataFrame, test_year: int, purge_days: int = 5):
    """
    Split data for walk-forward CV with purging.
    
    Train: All data from start through (test_year - 1)
    Purge: Remove last `purge_days` trading days from train
    Test: All data from test_year
    """
    # Test set: the entire test year
    test_mask = df['year'] == test_year
    
    # Train set: everything before test year
    train_mask = df['year'] < test_year
    
    # Get the last date in training set
    train_dates = df.loc[train_mask, 'date']
    if len(train_dates) == 0:
        return None, None, None, None
    
    max_train_date = train_dates.max()
    
    # Purge: remove last N trading days from train to avoid lookahead
    # Get unique dates and find cutoff
    unique_train_dates = sorted(train_dates.unique())
    if len(unique_train_dates) <= purge_days:
        return None, None, None, None
    
    purge_cutoff = unique_train_dates[-(purge_days + 1)]
    
    # Final train mask: before purge cutoff
    train_mask_purged = (df['year'] < test_year) & (df['date'] <= purge_cutoff)
    
    # Extract data
    train_df = df.loc[train_mask_purged].copy()
    test_df = df.loc[test_mask].copy()
    
    return train_df, test_df, purge_cutoff, max_train_date


def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Information Coefficient (Spearman correlation)."""
    from scipy.stats import spearmanr
    
    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic


def run_walk_forward_cv(df: pd.DataFrame, output_dir: Path):
    """Run purged walk-forward cross-validation."""
    
    results = []
    
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD CROSS-VALIDATION")
    logger.info("=" * 70)
    
    for test_year in CV_TEST_YEARS:
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST YEAR: {test_year}")
        logger.info(f"{'='*50}")
        
        # Get train/test split
        train_df, test_df, purge_cutoff, max_train_date = get_train_test_split(
            df, test_year, PURGE_DAYS
        )
        
        if train_df is None or len(train_df) < 1000:
            logger.warning(f"Insufficient training data for {test_year}, skipping")
            continue
        
        logger.info(f"Train: {len(train_df):,} rows (through {purge_cutoff.date()})")
        logger.info(f"Purged: {PURGE_DAYS} days (up to {max_train_date.date()})")
        logger.info(f"Test: {len(test_df):,} rows")
        
        # Prepare features
        X_train = train_df[FEATURES].copy()
        X_test = test_df[FEATURES].copy()
        
        y_train_reg = train_df[TARGET_REG].values
        y_test_reg = test_df[TARGET_REG].values
        
        y_train_clf = train_df[TARGET_CLF].values
        y_test_clf = test_df[TARGET_CLF].values
        
        # Handle missing values - fill with median from training set
        for col in FEATURES:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        
        # =====================================================================
        # REGRESSION MODEL
        # =====================================================================
        logger.info("\nTraining regression model...")
        
        # Create validation set from last 10% of training for early stopping
        n_train = len(X_train)
        n_val = int(n_train * 0.1)
        
        X_train_fit = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_train_fit = y_train_reg[:-n_val]
        y_val = y_train_reg[-n_val:]
        
        model_reg = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_REG
        )
        
        model_reg.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict on test
        y_pred_reg = model_reg.predict(X_test)
        
        # Calculate IC
        ic_reg = calculate_ic(y_test_reg, y_pred_reg)
        logger.info(f"Regression IC: {ic_reg:.4f}")
        
        # =====================================================================
        # CLASSIFICATION MODEL
        # =====================================================================
        logger.info("Training classification model...")
        
        y_train_clf_fit = y_train_clf[:-n_val]
        y_val_clf = y_train_clf[-n_val:]
        
        model_clf = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **XGB_PARAMS_CLF
        )
        
        model_clf.fit(
            X_train_fit, y_train_clf_fit,
            eval_set=[(X_val, y_val_clf)],
            verbose=False
        )
        
        # Predict probabilities on test
        y_pred_clf = model_clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        ic_clf = calculate_ic(y_test_reg, y_pred_clf)  # IC vs actual returns
        auc_clf = roc_auc_score(y_test_clf, y_pred_clf)
        
        logger.info(f"Classification IC: {ic_clf:.4f}")
        logger.info(f"Classification AUC: {auc_clf:.4f}")
        
        # Compare to linear baseline (alpha_composite_v8)
        if 'alpha_composite_v8' in test_df.columns:
            baseline_ic = calculate_ic(y_test_reg, test_df['alpha_composite_v8'].values)
            logger.info(f"Baseline v8 IC: {baseline_ic:.4f}")
        else:
            baseline_ic = np.nan
        
        # Store results
        results.append({
            'test_year': test_year,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'ic_regression': ic_reg,
            'ic_classification': ic_clf,
            'auc_classification': auc_clf,
            'ic_baseline_v8': baseline_ic,
            'n_estimators_reg': model_reg.best_iteration if hasattr(model_reg, 'best_iteration') else N_ESTIMATORS,
            'n_estimators_clf': model_clf.best_iteration if hasattr(model_clf, 'best_iteration') else N_ESTIMATORS,
        })
    
    # Save CV results
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'cv_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nCV results saved to {results_path}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("CV SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    logger.info(f"\nMean IC (Regression):     {results_df['ic_regression'].mean():.4f}")
    logger.info(f"Mean IC (Classification): {results_df['ic_classification'].mean():.4f}")
    logger.info(f"Mean IC (Baseline v8):    {results_df['ic_baseline_v8'].mean():.4f}")
    
    return results_df


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_models(df: pd.DataFrame, output_dir: Path):
    """Train final models on all available data."""
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL MODELS ON ALL DATA")
    logger.info("=" * 70)
    
    # Prepare data
    X = df[FEATURES].copy()
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values
    
    # Fill missing values with median
    medians = {}
    for col in FEATURES:
        medians[col] = X[col].median()
        X[col] = X[col].fillna(medians[col])
    
    # Save medians for production use
    medians_path = output_dir / 'feature_medians.json'
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=2)
    logger.info(f"Feature medians saved to {medians_path}")
    
    # Split for early stopping (use last 10% as validation)
    n = len(X)
    n_val = int(n * 0.1)
    
    X_train = X.iloc[:-n_val]
    X_val = X.iloc[-n_val:]
    y_train_reg = y_reg[:-n_val]
    y_val_reg = y_reg[-n_val:]
    y_train_clf = y_clf[:-n_val]
    y_val_clf = y_clf[-n_val:]
    
    # Train regression model
    logger.info(f"\nTraining final regression model on {len(X_train):,} rows...")
    
    model_reg = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS_REG
    )
    
    model_reg.fit(
        X_train, y_train_reg,
        eval_set=[(X_val, y_val_reg)],
        verbose=False
    )
    
    # Save regression model
    reg_path = output_dir / 'model_regression.json'
    model_reg.save_model(str(reg_path))
    logger.info(f"Regression model saved to {reg_path}")
    logger.info(f"  Best iteration: {model_reg.best_iteration}")
    
    # Train classification model
    logger.info(f"\nTraining final classification model...")
    
    model_clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS_CLF
    )
    
    model_clf.fit(
        X_train, y_train_clf,
        eval_set=[(X_val, y_val_clf)],
        verbose=False
    )
    
    # Save classification model
    clf_path = output_dir / 'model_classification.json'
    model_clf.save_model(str(clf_path))
    logger.info(f"Classification model saved to {clf_path}")
    logger.info(f"  Best iteration: {model_clf.best_iteration}")
    
    return model_reg, model_clf, medians


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def generate_shap_analysis(model_reg, model_clf, df: pd.DataFrame, output_dir: Path):
    """Generate SHAP feature importance plots."""
    
    if not HAS_SHAP:
        logger.warning("SHAP not installed, skipping feature importance plots")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("SHAP FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 70)
    
    # Sample data for SHAP (too slow on full dataset)
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample = sample_df[FEATURES].copy()
    for col in FEATURES:
        X_sample[col] = X_sample[col].fillna(X_sample[col].median())
    
    # Regression model SHAP
    logger.info("Computing SHAP values for regression model...")
    
    try:
        explainer_reg = shap.TreeExplainer(model_reg)
        shap_values_reg = explainer_reg.shap_values(X_sample)
        
        # Summary plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_reg, X_sample, show=False)
        plt.title("SHAP Feature Importance - Regression Model")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression.png', dpi=150)
        plt.close()
        logger.info(f"Saved SHAP plot: {output_dir / 'shap_regression.png'}")
        
        # Feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_reg, X_sample, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Mean |SHAP|) - Regression")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_regression_bar.png', dpi=150)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating regression SHAP: {e}")
    
    # Classification model SHAP
    logger.info("Computing SHAP values for classification model...")
    
    try:
        explainer_clf = shap.TreeExplainer(model_clf)
        shap_values_clf = explainer_clf.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_clf, X_sample, show=False)
        plt.title("SHAP Feature Importance - Classification Model")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_classification.png', dpi=150)
        plt.close()
        logger.info(f"Saved SHAP plot: {output_dir / 'shap_classification.png'}")
        
    except Exception as e:
        logger.error(f"Error generating classification SHAP: {e}")


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(
    con: duckdb.DuckDBPyConnection,
    model_reg,
    model_clf,
    medians: dict,
    output_dir: Path
):
    """Generate predictions for all rows and save to DuckDB."""
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING PREDICTIONS FOR ALL DATA")
    logger.info("=" * 70)
    
    # Load all data (including dates without targets for production)
    cols = FEATURES + ['ticker', 'date']
    cols_str = ', '.join(cols)
    
    logger.info("Loading full feature matrix...")
    df = con.execute(f"""
        SELECT {cols_str}
        FROM feat_matrix_v2
        WHERE date >= '2015-01-01'
    """).fetchdf()
    
    logger.info(f"Loaded {len(df):,} rows for prediction")
    
    # Prepare features
    X = df[FEATURES].copy()
    
    # Fill missing values with training medians
    for col in FEATURES:
        X[col] = X[col].fillna(medians.get(col, 0))
    
    # Generate predictions
    logger.info("Generating regression predictions...")
    df['alpha_ml_regression'] = model_reg.predict(X)
    
    logger.info("Generating classification predictions...")
    df['alpha_ml_classification'] = model_clf.predict_proba(X)[:, 1]
    
    # Keep only output columns
    output_df = df[['ticker', 'date', 'alpha_ml_regression', 'alpha_ml_classification']]
    
    # Save to DuckDB
    logger.info("Saving to feat_alpha_ml_xgb table...")
    
    con.execute("DROP TABLE IF EXISTS feat_alpha_ml_xgb")
    con.register("predictions_df", output_df)
    con.execute("""
        CREATE TABLE feat_alpha_ml_xgb AS 
        SELECT * REPLACE (CAST(date AS DATE) AS date)
        FROM predictions_df
    """)
    
    # Verify
    count = con.execute("SELECT COUNT(*) FROM feat_alpha_ml_xgb").fetchone()[0]
    logger.info(f"Created feat_alpha_ml_xgb with {count:,} rows")
    
    # Also save as parquet for backup
    parquet_path = output_dir / 'predictions.parquet'
    output_df.to_parquet(parquet_path, index=False)
    logger.info(f"Predictions also saved to {parquet_path}")
    
    return output_df


# =============================================================================
# VALIDATION: COMPARE TO BASELINE
# =============================================================================

def validate_vs_baseline(con: duckdb.DuckDBPyConnection):
    """Compare ML predictions to baseline alpha_composite_v8."""
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION: ML vs BASELINE")
    logger.info("=" * 70)
    
    comparison = con.execute("""
        WITH combined AS (
            SELECT 
                m.ticker,
                m.date,
                m.alpha_composite_v8,
                ml.alpha_ml_regression,
                ml.alpha_ml_classification,
                m.ret_5d_f
            FROM feat_matrix_v2 m
            JOIN feat_alpha_ml_xgb ml ON m.ticker = ml.ticker AND m.date = ml.date
            WHERE m.date >= '2015-01-01'
              AND m.ret_5d_f IS NOT NULL
              AND m.alpha_composite_v8 IS NOT NULL
        )
        SELECT 
            CORR(alpha_composite_v8, ret_5d_f) as ic_baseline,
            CORR(alpha_ml_regression, ret_5d_f) as ic_regression,
            CORR(alpha_ml_classification, ret_5d_f) as ic_classification,
            COUNT(*) as n_obs
        FROM combined
    """).fetchone()
    
    logger.info(f"\nOverall IC Comparison:")
    logger.info(f"  Baseline (v8):    {comparison[0]:.4f}")
    logger.info(f"  ML Regression:    {comparison[1]:.4f}")
    logger.info(f"  ML Classification: {comparison[2]:.4f}")
    logger.info(f"  (n = {comparison[3]:,})")
    
    # By year
    logger.info("\nIC by Year:")
    yearly = con.execute("""
        WITH combined AS (
            SELECT 
                YEAR(m.date) as year,
                m.alpha_composite_v8,
                ml.alpha_ml_regression,
                ml.alpha_ml_classification,
                m.ret_5d_f
            FROM feat_matrix_v2 m
            JOIN feat_alpha_ml_xgb ml ON m.ticker = ml.ticker AND m.date = ml.date
            WHERE m.date >= '2015-01-01'
              AND m.ret_5d_f IS NOT NULL
              AND m.alpha_composite_v8 IS NOT NULL
        )
        SELECT 
            year,
            CORR(alpha_composite_v8, ret_5d_f) as ic_v8,
            CORR(alpha_ml_regression, ret_5d_f) as ic_reg,
            CORR(alpha_ml_classification, ret_5d_f) as ic_clf,
            COUNT(*) as n
        FROM combined
        GROUP BY year
        ORDER BY year
    """).fetchdf()
    
    logger.info(f"\n{'Year':<6} {'IC_v8':>8} {'IC_Reg':>8} {'IC_Clf':>8} {'N':>10}")
    logger.info("-" * 45)
    for _, row in yearly.iterrows():
        logger.info(f"{int(row['year']):<6} {row['ic_v8']:>8.4f} {row['ic_reg']:>8.4f} {row['ic_clf']:>8.4f} {int(row['n']):>10,}")
    
    # Correlation between ML and baseline
    corr = con.execute("""
        SELECT 
            CORR(alpha_composite_v8, alpha_ml_regression) as corr_v8_reg,
            CORR(alpha_composite_v8, alpha_ml_classification) as corr_v8_clf,
            CORR(alpha_ml_regression, alpha_ml_classification) as corr_reg_clf
        FROM feat_matrix_v2 m
        JOIN feat_alpha_ml_xgb ml ON m.ticker = ml.ticker AND m.date = ml.date
        WHERE m.date >= '2015-01-01'
          AND m.alpha_composite_v8 IS NOT NULL
    """).fetchone()
    
    logger.info(f"\nSignal Correlations:")
    logger.info(f"  v8 vs ML Regression:     {corr[0]:.3f}")
    logger.info(f"  v8 vs ML Classification: {corr[1]:.3f}")
    logger.info(f"  Regression vs Classification: {corr[2]:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost alpha models")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="scripts/ml/outputs", help="Output directory")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis (faster)")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV, only train final models")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("XGBOOST ALPHA MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Features: {FEATURES}")
    
    # Connect to database
    con = duckdb.connect(args.db)
    
    try:
        # Load data
        df = load_data(con)
        
        # Also load alpha_composite_v8 for comparison
        v8_data = con.execute("""
            SELECT ticker, date, alpha_composite_v8
            FROM feat_matrix_v2
            WHERE date >= '2015-01-01'
        """).fetchdf()
        v8_data['date'] = pd.to_datetime(v8_data['date'])
        
        df = df.merge(v8_data, on=['ticker', 'date'], how='left')
        
        # Run walk-forward CV
        if not args.skip_cv:
            cv_results = run_walk_forward_cv(df, output_dir)
        
        # Train final models
        model_reg, model_clf, medians = train_final_models(df, output_dir)
        
        # SHAP analysis
        if not args.skip_shap and HAS_SHAP:
            generate_shap_analysis(model_reg, model_clf, df, output_dir)
        
        # Generate predictions and save to DuckDB
        predictions = generate_predictions(con, model_reg, model_clf, medians, output_dir)
        
        # Validate vs baseline
        validate_vs_baseline(con)
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nOutputs saved to: {output_dir}")
        logger.info("  - cv_results.csv")
        logger.info("  - model_regression.json")
        logger.info("  - model_classification.json")
        logger.info("  - feature_medians.json")
        logger.info("  - predictions.parquet")
        if not args.skip_shap and HAS_SHAP:
            logger.info("  - shap_regression.png")
            logger.info("  - shap_classification.png")
        logger.info("\nDuckDB table created: feat_alpha_ml_xgb")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()