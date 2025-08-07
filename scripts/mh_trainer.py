#!/usr/bin/env python3
import argparse
import duckdb
import pandas as pd
import numpy as np
import joblib

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward trainer for multi-horizon windows"
    )
    parser.add_argument(
        "--db-path", default="data/kairos.duckdb",
        help="Path to DuckDB file containing mh_windows table"
    )
    parser.add_argument(
        "--freq", default="Q",
        help="Fold frequency for retrain (e.g. 'Q' for quarterly, 'A' for annual)"
    )
    parser.add_argument(
        "--window-size", type=int, default=252,
        help="Input window length"
    )
    parser.add_argument(
        "--max-horizon", type=int, default=21,
        help="Max forward horizon for targets"
    )
    parser.add_argument(
        "--embargo", type=int,
        default=None,
        help="Embargo days between train and validation splits (defaults to max_horizon)"
    )
    parser.add_argument(
        "--model-out", default="models/mh_model_final.pkl",
        help="Path to save final retrained model"
    )
    return parser.parse_args()


def get_fold_dates(conn, table: str, freq: str, window_size: int, max_horizon: int):
    """
    Generate fold end dates by snapping boundaries (e.g. quarter-ends) to actual trading days.
    """
    df = conn.execute(f"""
        SELECT DISTINCT CAST(end_date AS DATE) AS d
        FROM {table}
    """
    ).df().sort_values("d")
    all_dates = df["d"].reset_index(drop=True)

    # first valid date index to ensure look-back
    first_idx = window_size + max_horizon - 1
    first_fold = all_dates.iloc[first_idx]

    # theoretical boundaries
    boundaries = pd.date_range(start=first_fold, end=all_dates.iloc[-1], freq=freq)

    # snap to actual trading days <= boundary
    fold_dates = []
    for b in boundaries:
        real = all_dates[all_dates <= b].max()
        if not fold_dates or real > fold_dates[-1]:
            fold_dates.append(real)

    # ensure final date included
    last = all_dates.iloc[-1]
    if fold_dates[-1] != last:
        fold_dates.append(last)

    return fold_dates


def load_windows(conn, table: str, end_date_max, end_date_min=None):
    """
    Load feature arrays and target matrix for windows:
    end_date > end_date_min (if set) and <= end_date_max.
    Returns X [n_samples, features], y [n_samples, horizons].
    """
    where = []
    if end_date_min:
        where.append(f"end_date > DATE '{end_date_min}'")
    where.append(f"end_date <= DATE '{end_date_max}'")
    clause = " AND ".join(where)

    df = conn.execute(f"""
        SELECT features, ret_1d_f, ret_5d_f, ret_21d_f
        FROM {table}
        WHERE {clause}
    """
    ).df()

    if df.empty:
        return np.empty((0,)), np.empty((0,3))

    X = np.vstack(df['features'].to_list())
    y = df[['ret_1d_f','ret_5d_f','ret_21d_f']].values
    return X, y


def main():
    args = parse_args()
    embargo_days = args.embargo if args.embargo is not None else args.max_horizon

    # single connection for both read and write
    conn = duckdb.connect(database=args.db_path)

    # create metrics table if not exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mh_fold_metrics (
      fold      INTEGER,
      train_end DATE,
      val_end   DATE,
      acc_1d    DOUBLE,
      mse_1d    DOUBLE
    )""")

    # generate folds
    folds = get_fold_dates(
        conn,
        table="mh_windows",
        freq=args.freq,
        window_size=args.window_size,
        max_horizon=args.max_horizon
    )
    print("Fold dates:")
    for d in folds:
        print("  ", d)

    # walk-forward training and validation
    for i in range(len(folds)-1):
        D_train = folds[i]
        D_val   = folds[i+1]
        embargo_date = (pd.Timestamp(D_train) + pd.Timedelta(days=embargo_days)).date()

        print(f"=== Fold {i+1}: Train<= {D_train}, Embargo<= {embargo_date}, Validate<= {D_val} ===")

        # load data with embargo
        X_train, y_train = load_windows(
            conn,
            'mh_windows',
            end_date_max=D_train
        )
        X_val, y_val = load_windows(
            conn,
            'mh_windows',
            end_date_max=D_val,
            end_date_min=embargo_date
        )

        print(f"Training on {X_train.shape[0]} windows; validating on {X_val.shape[0]} windows.")

        # train model
        model = MultiOutputRegressor(RandomForestRegressor(n_jobs=-1))
        model.fit(X_train, y_train)

        # predict and score
        y_pred = model.predict(X_val)
        acc_1d = (np.sign(y_pred[:,0]) == np.sign(y_val[:,0])).mean() # type: ignore
        mse_1d = mean_squared_error(y_val[:,0], y_pred[:,0]) # type: ignore
        print(f"Fold {i+1} — 1d accuracy: {acc_1d:.4f}, 1d MSE: {mse_1d:.6f}")

        # persist metrics
        conn.execute(
            "INSERT INTO mh_fold_metrics VALUES (?, ?, ?, ?, ?)",
            (i+1, D_train, D_val, float(acc_1d), float(mse_1d))
        )

    # final retrain on all data up to last fold
    print("Retraining on all data up to last fold and saving final model...")
    model_final = MultiOutputRegressor(RandomForestRegressor(n_jobs=-1))
    X_all, y_all = load_windows(
        conn,
        'mh_windows',
        end_date_max=folds[-1]
    )
    model_final.fit(X_all, y_all)
    joblib.dump(model_final, args.model_out)
    print(f"Saved final model to {args.model_out}")

    conn.close()
    print("✅ Walk-forward training complete. Metrics in mh_fold_metrics.")

if __name__ == "__main__":
    main()
