#!/usr/bin/env python3
"""
Generate per-ticker multi-horizon dataset shards from DuckDB.
Supports stride=1, window sizes of 126 or 252.
Each ticker outputs 3 files:
  mh_<window>_<TICKER>_X.parquet
  mh_<window>_<TICKER>_y.parquet
  mh_<window>_<TICKER>_meta.parquet

Options:
  --window <int>        : Size of rolling input window (default: 126)
  --n_jobs <int>        : Parallel workers (0 = all cores)
  --full                : Use full feature set from feat_matrix
  --out-dir <path>      : Where to save shards (default: data/shards/)
"""

import os
import sys
import argparse
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler

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

        try:
            last_price = hist["closeadj"].iloc[-1]

            # Defensive float conversion
            ret_1d = float(np.log(fwd["closeadj"].iloc[1] / last_price))
            ret_5d = float(np.log(fwd["closeadj"].iloc[5] / last_price))
            ret_21d = float(np.log(fwd["closeadj"].iloc[21] / last_price))

            X_list.append(hist[feature_cols].values.flatten())
            y_list.append([ret_1d, ret_5d, ret_21d])
            meta_records.append({
                "ticker": hist["ticker"].iloc[0],
                "start_date": hist["date"].iloc[0],
                "end_date": hist["date"].iloc[-1]
            })
        except Exception:
            continue  # skip bad frame

    return X_list, y_list, meta_records


def process_ticker(ticker: str, window: int, full: bool, out_dir: str):
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        if full:
            df = con.execute(f"""
                SELECT fm.*, sb.closeadj
                FROM feat_matrix fm
                JOIN sep_base sb USING (ticker, date)
                WHERE ticker = '{ticker}' ORDER BY date
            """).df()
            feature_cols = [col for col in df.columns if col not in ("ticker", "date", "closeadj")]
        else:
            df = con.execute(f"""
                SELECT date, ticker, closeadj, volume
                FROM mid_cap_2025_07_15
                WHERE ticker = '{ticker}' ORDER BY date
            """).df()
            df = compute_basic_features(df)
            feature_cols = ["log_return", "volume_z", "price_norm"]

        if len(df) < window + 21:
            return

        X_list, y_list, meta_records = create_windows(df, window, feature_cols)
        if not X_list:
            return

        prefix = os.path.join(out_dir, f"mh_{window}_{ticker}")
        pd.DataFrame(X_list).to_parquet(f"{prefix}_X.parquet", index=False)
        pd.DataFrame(y_list, columns=["ret_1d_f", "ret_5d_f", "ret_21d_f"]).to_parquet(f"{prefix}_y.parquet", index=False)
        pd.DataFrame(meta_records).to_parquet(f"{prefix}_meta.parquet", index=False)

    except Exception as e:
        print(f"[{ticker}] Failed: {e}", file=sys.stderr)
    finally:
        try:
            con.close()
        except:
            pass


def build_shards(window: int, n_jobs: int, full: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=True)
    tickers = [row[0] for row in con.execute(
        "SELECT DISTINCT ticker FROM feat_matrix" if full else
        "SELECT DISTINCT ticker FROM mid_cap_2025_07_15"
    ).fetchall()]
    con.close()

    print(f"⏳ Building shards with window={window}, full={full}, on {len(tickers)} tickers...")

    max_workers = None if n_jobs <= 0 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_ticker, t, window, full, out_dir): t for t in tickers
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tickers"):
            try:
                future.result()
            except Exception as e:
                print(str(e), file=sys.stderr)

    print(f"✅ Shards saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, choices=[126, 252], default=126,
                        help="Rolling window size (126 or 252)")
    parser.add_argument("--n_jobs", type=int, default=0,
                        help="Number of parallel workers (0 = all cores)")
    parser.add_argument("--full", action="store_true",
                        help="Use full feature set from feat_matrix")
    parser.add_argument("--out-dir", type=str, default="data/shards/",
                        help="Directory to write output shards")
    args = parser.parse_args()

    build_shards(args.window, args.n_jobs, args.full, args.out_dir)
