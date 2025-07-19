#!/usr/bin/env python3
"""
Generate multi-horizon dataset from DuckDB tables.
Supports rolling window sizes of 252 or 126.
Parallel processing with ProcessPoolExecutor and progress bar via tqdm.
Writes output to DuckDB tables: mh_X_<window>, mh_y_<window>, mh_meta_<window>
Cleans up intermediate Parquet shards after table creation.

Add `--full` to use full 75+ feature matrix from `feat_matrix` table (built via build_full_feature_matrix.py).
"""

import argparse
import sys
import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import glob
import os

# Path to DuckDB database
DB_PATH = "data/kairos.duckdb"


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")
    df["log_return"] = np.log(df["closeadj"] / df["closeadj"].shift(1))
    df["volume_z"] = StandardScaler().fit_transform(df[["volume"]])
    df["price_norm"] = StandardScaler().fit_transform(df[["closeadj"]])
    return df.dropna()


def create_windows(df: pd.DataFrame, window: int, feature_cols: list):
    X_list, y_list, meta_records = [], [], []
    total = len(df)
    for i in range(total - window - 21):
        hist = df.iloc[i : i + window]
        fwd = df.iloc[i + window : i + window + 22]
        if len(hist) < window or len(fwd) < 22:
            continue
        last_price = hist["closeadj"].iloc[-1]
        # Flatten selected features
        X_list.append(hist[feature_cols].values.flatten())
        # Multi-horizon targets: 1d, 5d, 21d log returns
        y_list.append([
            np.log(fwd["closeadj"].iloc[1] / last_price),
            np.log(fwd["closeadj"].iloc[5] / last_price),
            np.log(fwd["closeadj"].iloc[21] / last_price)
        ])
        meta_records.append({
            "ticker": hist["ticker"].iloc[0],
            "start_date": hist["date"].iloc[0],
            "end_date": hist["date"].iloc[-1]
        })
    return X_list, y_list, meta_records


def process_ticker(ticker: str, window: int, full: bool):
    con = duckdb.connect(DB_PATH)
    try:
        if full:
            # Use full feature matrix plus closeadj from sep_base
            df = con.execute(
                f"""
                SELECT fm.*, sb.closeadj
                FROM feat_matrix fm
                JOIN sep_base sb USING (ticker, date)
                WHERE ticker = '{ticker}'
                ORDER BY date
                """
            ).df()
            feature_cols = [c for c in df.columns if c not in ("ticker", "date", "closeadj")]
        else:
            # Basic price-action features
            df = con.execute(
                f"SELECT date, ticker, closeadj, volume FROM mid_cap_2025_07_15 "
                f"WHERE ticker = '{ticker}' ORDER BY date"
            ).df()
            df = compute_basic_features(df)
            feature_cols = ["log_return", "volume_z", "price_norm"]

        if len(df) < window + 21:
            return

        X_list, y_list, meta_records = create_windows(df, window, feature_cols)
        if not X_list:
            return

        # Write shards
        X_df = pd.DataFrame(X_list)
        y_df = pd.DataFrame(y_list, columns=["ret_1d_f", "ret_5d_f", "ret_21d_f"])
        meta_df = pd.DataFrame(meta_records)
        prefix = f"mh_{window}_{ticker}"
        X_df.to_parquet(f"{prefix}_X.parquet", index=False)
        y_df.to_parquet(f"{prefix}_y.parquet", index=False)
        meta_df.to_parquet(f"{prefix}_meta.parquet", index=False)
    finally:
        con.close()


def build_mh_dataset(window: int, n_jobs: int, full: bool):
    master_con = duckdb.connect(DB_PATH)
    tickers = [row[0] for row in master_con.execute(
        "SELECT DISTINCT ticker FROM mid_cap_2025_07_15" if not full else
        "SELECT DISTINCT ticker FROM feat_matrix"
    ).fetchall()]
    print(f"Building multi-horizon dataset (window={window}, full={full}) on {len(tickers)} tickers using {'all' if n_jobs<=0 else n_jobs} cores...")

    # Remove old shards
    for shard in glob.glob(f"mh_{window}_*_*.parquet"):
        try:
            os.remove(shard)
        except OSError:
            print(f"Warning: Could not remove shard {shard}", file=sys.stderr)
            pass

    # Parallel execution
    max_workers = None if n_jobs <= 0 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_ticker, t, window, full): t for t in tickers
        }
        for _ in tqdm(futures, total=len(tickers), desc="Processing tickers"):
            pass

    # Combine shards into DuckDB tables
    for name in ["X", "y", "meta"]:
        table = f"mh_{name}_{window}"
        pattern = f"mh_{window}_*_{name}.parquet"
        master_con.execute(f"DROP TABLE IF EXISTS {table}")
        master_con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{pattern}')"
        )
        # Clean up shards
        for shard in glob.glob(pattern):
            try:
                os.remove(shard)
            except OSError:
                pass

    print(f"✅ Built tables in DuckDB: mh_X_{window}, mh_y_{window}, mh_meta_{window}")
    master_con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-horizon dataset with full-feature option."
    )
    parser.add_argument(
        "--window", type=int, choices=[126, 252], default=252,
        help="Rolling window size in trading days"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=0,
        help="Number of parallel workers; 0 or negative uses all cores"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full 75+ feature matrix from feat_matrix table"
    )
    args = parser.parse_args()
    build_mh_dataset(args.window, args.n_jobs, args.full)
