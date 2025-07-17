#!/usr/bin/env python3
"""
Train a multi-horizon LightGBM model on mh_X_<window> / mh_y_<window> tables.
Evaluates MSE, R² and directional accuracy, then saves the model.
"""

import argparse
import duckdb
import joblib
import numpy as np
from typing import cast
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

def load_data(db_path: str, window: int):
    con = duckdb.connect(db_path)
    X = con.execute(f"SELECT * FROM mh_X_{window}").df().values
    y = con.execute(f"SELECT * FROM mh_y_{window}").df().values
    con.close()
    return X, y

def evaluate(y_true, y_pred):
    horizons = ['1d','5d','21d']
    for i, h in enumerate(horizons):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2  = r2_score(y_true[:, i], y_pred[:, i])
        acc = (np.sign(y_pred[:, i]) == np.sign(y_true[:, i])).mean() * 100
        print(f"{h}: MSE={mse:.5f}, R²={r2:.3f}, DirAcc={acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Train multi-horizon LGBM model")
    parser.add_argument("--window", type=int, choices=[126,252], default=252)
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-out", type=str, default="mh_lgbm.pkl")
    args = parser.parse_args()

    # 1) Load & split
    X, y = load_data(args.db, args.window)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, shuffle=True, random_state=42
    )

    # 2) Build & train multi-output model
    # Cast LGBMRegressor to BaseEstimator to satisfy type checker
    base_est = cast(
        BaseEstimator,
        LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves
        )
    )
    model = MultiOutputRegressor(base_est)
    model.fit(X_train, y_train)

    # 3) Evaluate
    print("\nValidation performance:")
    y_pred = model.predict(X_val)
    evaluate(y_val, y_pred)

    # 4) Save
    joblib.dump(model, args.model_out)
    print(f"\nModel saved to {args.model_out}")

if __name__ == "__main__":
    main()
