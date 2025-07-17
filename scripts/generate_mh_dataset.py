#!/usr/bin/env python3
"""
Generate multi-horizon dataset from mid_cap_2025_07_15 DuckDB table in the goldenDuck database.
Supports rolling window sizes of 252 or 126.
Parallel processing with ProcessPoolExecutor and progress bar via tqdm.
Writes output to DuckDB tables: mh_X_<window>, mh_y_<window>, mh_meta_<window>
Cleans up intermediate Parquet shards after table creation.
"""

import argparse
import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import os

# Path to DuckDB database (your "goldenDuck")
db_path = "data/kairos.duckdb"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")
    df["log_return"] = np.log(df["closeadj"] / df["closeadj"].shift(1))
    df["volume_z"] = StandardScaler().fit_transform(df[["volume"]])
    df["price_norm"] = StandardScaler().fit_transform(df[["closeadj"]])
    return df.dropna()


def create_windows(df: pd.DataFrame, window: int):
    X_list, y_list, meta_records = [], [], []
    total = len(df)
    for i in range(total - window - 21):
        hist = df.iloc[i : i + window]
        fwd = df.iloc[i + window : i + window + 22]
        if len(hist) < window or len(fwd) < 22:
            continue
        last_price = hist["closeadj"].iloc[-1]
        X_list.append(hist[["log_return", "volume_z", "price_norm"]].values.flatten())
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


def process_ticker(ticker: str, window: int):
    con = duckdb.connect(db_path)
    try:
        df = con.execute(
            f"SELECT date, ticker, closeadj, volume FROM mid_cap_2025_07_15 WHERE ticker = '{ticker}' ORDER BY date"
        ).df()
        if len(df) < window + 21:
            return
        df_feat = compute_features(df)
        X_list, y_list, meta_records = create_windows(df_feat, window)
        if not X_list:
            return
        X_df = pd.DataFrame(X_list)
        y_df = pd.DataFrame(y_list, columns=["ret_1d_f", "ret_5d_f", "ret_21d_f"])
        meta_df = pd.DataFrame(meta_records)
        prefix = f"mh_{window}_{ticker}"
        X_df.to_parquet(f"{prefix}_X.parquet", index=False)
        y_df.to_parquet(f"{prefix}_y.parquet", index=False)
        meta_df.to_parquet(f"{prefix}_meta.parquet", index=False)
    finally:
        con.close()


def build_mh_dataset(window: int, n_jobs: int):
    master_con = duckdb.connect(db_path)
    tickers = [row[0] for row in master_con.execute(
        "SELECT DISTINCT ticker FROM mid_cap_2025_07_15").fetchall()]
    print(f"Building multi-horizon dataset (window={window}) on {len(tickers)} tickers using {n_jobs if n_jobs>0 else 'all'} cores...")

    # Remove old shard files
    for shard in glob.glob(f"mh_{window}_*_*.parquet"):
        try:
            os.remove(shard)
        except OSError:
            pass

    # Parallel execution with progress bar
    max_workers = None if n_jobs <= 0 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ticker, t, window): t for t in tickers}
        for _ in tqdm(as_completed(futures), total=len(tickers), desc="Processing tickers"):
            pass

    # Combine shards into DuckDB tables
    for name in ["X", "y", "meta"]:
        table = f"mh_{name}_{window}"
        pattern = f"mh_{window}_*_{name}.parquet"
        master_con.execute(f"DROP TABLE IF EXISTS {table}")
        master_con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{pattern}')")
        # Clean up shards for this table
        for shard in glob.glob(pattern):
            try:
                os.remove(shard)
            except OSError:
                pass

    print(f"✅ Built tables in goldenDuck: mh_X_{window}, mh_y_{window}, mh_meta_{window}")
    master_con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-horizon dataset with parallelism, progress bar, and shard cleanup."
    )
    parser.add_argument(
        "--window", type=int, choices=[126, 252], default=252,
        help="Rolling window size in trading days"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=0,
        help="Number of parallel workers; 0 or negative uses all cores"
    )
    args = parser.parse_args()
    build_mh_dataset(args.window, args.n_jobs)
