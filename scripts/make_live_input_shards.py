#!/usr/bin/env python3
"""
make_live_input_shards.py

For each ticker in DuckDB:
- Extract the latest <window> days
- Join with feat_matrix if --full
- Save a single live input row per ticker to:
  live_<window>_<TICKER>_X.parquet
  live_<window>_<TICKER>_meta.parquet
"""

import os
import argparse
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DB_PATH = "data/kairos.duckdb"

def get_live_input(con, ticker: str, window: int, full: bool) -> tuple:
    if full:
        df = con.execute(f"""
            SELECT fm.*, sb.closeadj
            FROM feat_matrix fm
            JOIN sep_base sb USING (ticker, date)
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT {window}
        """).df()
        df = df.sort_values("date")
        feature_cols = [col for col in df.columns if col not in ("ticker", "date", "closeadj")]
    else:
        df = con.execute(f"""
            SELECT date, ticker, closeadj, volume
            FROM sep_base
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT {window}
        """).df()
        df = df.sort_values("date")
        df["log_return"] = np.log(df["closeadj"] / df["closeadj"].shift(1))
        df["volume_z"] = StandardScaler().fit_transform(df[["volume"]])
        df["price_norm"] = StandardScaler().fit_transform(df[["closeadj"]])
        feature_cols = ["log_return", "volume_z", "price_norm"]

    if len(df) < window:
        return None, None

    X = df[feature_cols].values.astype(np.float32).reshape(1, -1)
    meta = pd.DataFrame([{
        "ticker": ticker,
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1]
    }])
    return X, meta


def main(window: int, out_dir: str, full: bool):
    os.makedirs(out_dir, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    tickers = con.execute(
        "SELECT DISTINCT ticker FROM feat_matrix" if full else
        "SELECT DISTINCT ticker FROM sep_base"
    ).fetchall()
    tickers = sorted([t[0] for t in tickers])

    for ticker in tqdm(tickers, desc="Building live inputs"):
        try:
            X, meta = get_live_input(con, ticker, window, full)
            if X is None:
                continue
            prefix = os.path.join(out_dir, f"live_{window}_{ticker}")
            pd.DataFrame(X).to_parquet(prefix + "_X.parquet", index=False)
            meta.to_parquet(prefix + "_meta.parquet", index=False)
        except Exception as e:
            print(f"[{ticker}] ⚠️ Failed: {e}")
            continue

    print(f"✅ Live input shards written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=126, help="Window size (e.g., 126 or 252)")
    parser.add_argument("--out-dir", type=str, default="data/live_inputs/", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Use full feat_matrix feature set")
    args = parser.parse_args()

    main(args.window, args.out_dir, args.full)
