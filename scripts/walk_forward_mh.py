#!/usr/bin/env python3
"""
walk_forward_mh.py

Perform walk-forward (rolling) validation on your multi-horizon dataset.
For each test year T in [initial_train_year+1 … last_year]:
  • Train on all data with start_date.year <= T-1
  • Test on data with start_date.year == T
Aggregate MSE, R², and directional accuracy across all folds.
"""

import argparse
import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from typing import Tuple, List, Dict

def load_data(db_path: str, window: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    con = duckdb.connect(db_path)
    meta = con.execute(f"SELECT * FROM mh_meta_{window}").df()
    X_df = con.execute(f"SELECT * FROM mh_X_{window}").df()
    y_df = con.execute(f"SELECT * FROM mh_y_{window}").df()
    con.close()

    # ensure date is datetime and extract year
    meta['start_date'] = pd.to_datetime(meta['start_date'])
    meta['year'] = meta['start_date'].dt.year

    # align X, y with meta by row order
    X = X_df.values
    y = y_df.values
    return meta, X, y

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    horizons = ['1d','5d','21d']
    metrics = {}
    for i, h in enumerate(horizons):
        metrics[f'mse_{h}'] = mean_squared_error(y_true[:, i], y_pred[:, i])
        metrics[f'r2_{h}']  = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f'acc_{h}'] = (np.sign(y_pred[:, i]) == np.sign(y_true[:, i])).mean()
    return metrics

def walk_forward_validation(
    meta: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    initial_train_year: int,
    lgbm_params: Dict
) -> None:
    years = sorted(meta['year'].unique())
    test_years = [yr for yr in years if yr >= initial_train_year + 1]
    all_true, all_pred = [], []
    fold_metrics: List[Dict] = []

    print(f"Running walk-forward from train_end={initial_train_year} to last_year={years[-1]}")
    for test_year in test_years:
        train_mask = meta['year'] <= (test_year - 1)
        test_mask  = meta['year'] == test_year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask.values], y[train_mask.values]
        X_test,  y_test  = X[test_mask.values],  y[test_mask.values]

        # build & train
        base_est = LGBMRegressor(**lgbm_params)
        model = MultiOutputRegressor(base_est) # type: ignore
        model.fit(X_train, y_train)

        # predict & metric
        y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred) # type: ignore
        m['test_year'] = test_year
        fold_metrics.append(m)

        all_true.append(y_test)
        all_pred.append(y_pred)

        print(f"Year {test_year}: "
              f"MSE_1d={m['mse_1d']:.5f}, R2_1d={m['r2_1d']:.3f}, Acc_1d={m['acc_1d']:.3f} | "
              f"MSE_5d={m['mse_5d']:.5f}, R2_5d={m['r2_5d']:.3f}, Acc_5d={m['acc_5d']:.3f} | "
              f"MSE_21d={m['mse_21d']:.5f}, R2_21d={m['r2_21d']:.3f}, Acc_21d={m['acc_21d']:.3f}")

    # aggregate across all folds
    if all_true:
        y_true_all = np.vstack(all_true)
        y_pred_all = np.vstack(all_pred)
        agg = compute_metrics(y_true_all, y_pred_all)
        print("\n=== Aggregated Across All Folds ===")
        print(f"MSE_1d={agg['mse_1d']:.5f}, R2_1d={agg['r2_1d']:.3f}, Acc_1d={agg['acc_1d']:.3f}")
        print(f"MSE_5d={agg['mse_5d']:.5f}, R2_5d={agg['r2_5d']:.3f}, Acc_5d={agg['acc_5d']:.3f}")
        print(f"MSE_21d={agg['mse_21d']:.5f}, R2_21d={agg['r2_21d']:.3f}, Acc_21d={agg['acc_21d']:.3f}")
    else:
        print("No test folds were run (check your initial_train_year and data range).")

def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for multi-horizon model")
    parser.add_argument("--db",               type=str,   default="data/kairos.duckdb")
    parser.add_argument("--window",           type=int,   choices=[126,252], default=252)
    parser.add_argument("--initial-train-year", type=int, default=2015,
                        help="Last year of the first training period (e.g. 2015 → first test is 2016)")
    parser.add_argument("--n-estimators",     type=int,   default=500)
    parser.add_argument("--learning-rate",    type=float, default=0.05)
    parser.add_argument("--num-leaves",       type=int,   default=31)
    args = parser.parse_args()

    meta, X, y = load_data(args.db, args.window)
    lgbm_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves
    }
    walk_forward_validation(meta, X, y, args.initial_train_year, lgbm_params)

if __name__ == "__main__":
    main()
