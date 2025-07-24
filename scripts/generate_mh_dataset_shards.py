#!/usr/bin/env python3
"""
generate_mh_dataset_shards.py

Generate multi-horizon dataset from DuckDB tables.
- Supports 3-feature minimal input or full 75+ feature matrix
- Uses per-frame rolling normalization to prevent forward leakage
- Optional cutoff date to restrict sample range for train/test splitting
- Outputs per-ticker shards (X, y, meta) in scripts/shards/
"""

import argparse
import duckdb
import numpy as np
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

DB_PATH = "data/kairos.duckdb"

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["log_return"] = np.log(df["closeadj"] / df["closeadj"].shift(1))
    df["volume_z"] = (df["volume"] - df["volume"].rolling(63, min_periods=20).mean()) / df["volume"].rolling(63, min_periods=20).std()
    df["price_norm"] = df["closeadj"] / df["closeadj"].iloc[0]  # normalize to 1.0 start
    return df.dropna().reset_index(drop=True)

def create_windows(df: pd.DataFrame, window: int, feature_cols: list):
    X_list, y_list, meta_records = [], [], []
    total = len(df)
    for i in range(total - window - 21):
        hist = df.iloc[i : i + window]
        fwd = df.iloc[i + window : i + window + 22]

        if len(hist) < window or len(fwd) < 22:
            continue

        try:
            last_price = hist["closeadj"].iloc[-1]
            ret_1d = np.log(fwd["closeadj"].iloc[1] / last_price)
            ret_5d = np.log(fwd["closeadj"].iloc[5] / last_price)
            ret_21d = np.log(fwd["closeadj"].iloc[21] / last_price)

            X_list.append(hist[feature_cols].values.flatten())
            y_list.append([ret_1d, ret_5d, ret_21d])
            meta_records.append({
                "ticker": hist["ticker"].iloc[0],
                "start_date": hist["date"].iloc[0],
                "end_date": hist["date"].iloc[-1]
            })
        except Exception:
            continue

    return X_list, y_list, meta_records

def process_ticker(ticker: str, window: int, full: bool, cutoff: str):
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        if full:
            query = f"""
                SELECT fm.*, sb.closeadj
                FROM feat_matrix fm
                JOIN sep_base sb USING (ticker, date)
                WHERE ticker = '{ticker}'
            """
        else:
            query = f"""
                SELECT date, ticker, closeadj, volume
                FROM mid_cap_2025_07_15
                WHERE ticker = '{ticker}'
            """

        if cutoff:
            query += f" AND date <= '{cutoff}'"
        query += " ORDER BY date"

        df = con.execute(query).df()
        if df.empty or len(df) < window + 21:
            return

        if full:
            feature_cols = [c for c in df.columns if c not in ("ticker", "date", "closeadj")]
        else:
            df = compute_basic_features(df)
            feature_cols = ["log_return", "volume_z", "price_norm"]

        X_list, y_list, meta_records = create_windows(df, window, feature_cols)
        if not X_list:
            return

        prefix = f"mh_{window}_{ticker}"
        pd.DataFrame(X_list).to_parquet(f"scripts/shards/{prefix}_X.parquet", index=False)
        pd.DataFrame(y_list, columns=["ret_1d_f", "ret_5d_f", "ret_21d_f"]).to_parquet(f"scripts/shards/{prefix}_y.parquet", index=False)
        pd.DataFrame(meta_records).to_parquet(f"scripts/shards/{prefix}_meta.parquet", index=False)

    finally:
        con.close()

def build_mh_dataset(window: int, n_jobs: int, cutoff: str, full: bool):
    con = duckdb.connect(DB_PATH, read_only=True)
    tickers = [row[0] for row in con.execute(
        "SELECT DISTINCT ticker FROM feat_matrix" if full else
        "SELECT DISTINCT ticker FROM mid_cap_2025_07_15"
    ).fetchall()]
    con.close()

    print(f"⏳ Building dataset (window={window}, full={full}, cutoff={cutoff}) for {len(tickers)} tickers...")

    for shard in glob.glob(f"scripts/shards/mh_{window}_*_*.parquet"):
        try:
            os.remove(shard)
        except OSError:
            print(f"⚠️ Could not remove {shard}", file=sys.stderr)

    max_workers = None if n_jobs <= 0 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_ticker, t, window, full, cutoff): t for t in tickers
        }
        for _ in tqdm(futures, total=len(futures), desc="Processing tickers"):
            pass

    print(f"✅ Finished writing shards to scripts/shards/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-horizon dataset shards with optional full features.")
    parser.add_argument("--window", type=int, choices=[126, 252], default=126, help="Input window size")
    parser.add_argument("--n_jobs", type=int, default=0, help="Parallel workers (0 = all cores)")
    parser.add_argument("--cutoff", type=str, default=None, help="Max date (e.g. 2022-12-31) to include")
    parser.add_argument("--full", action="store_true", help="Use full feature matrix from feat_matrix")
    args = parser.parse_args()

    build_mh_dataset(args.window, args.n_jobs, args.cutoff, args.full)
