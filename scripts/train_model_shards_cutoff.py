#!/usr/bin/env python3
"""
train_model_shards_cutoff.py

Train a multi-output regression model from per-ticker shards using only rows
where meta['end_date'] <= cutoff date (e.g. 2022-12-31).

Each shard group:
    mh_<window>_<TICKER>_X.parquet
    mh_<window>_<TICKER>_y.parquet
    mh_<window>_<TICKER>_meta.parquet

Supports:
- XGBoost or Ridge
- Save every N tickers

Usage:
    python train_model_shards_cutoff.py \
        --window 126 \
        --model xgb \
        --cutoff 2022-12-31 \
        --out-dir scripts/shards \
        --save-every 10
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

def get_tickers(out_dir: str, window: int):
    x_files = glob.glob(os.path.join(out_dir, f"mh_{window}_*_X.parquet"))
    tickers = sorted([os.path.basename(f).split("_")[2] for f in x_files])
    return tickers

def load_and_filter_shard(ticker: str, window: int, out_dir: str, cutoff: pd.Timestamp):
    prefix = os.path.join(out_dir, f"mh_{window}_{ticker}")
    try:
        meta = pd.read_parquet(prefix + "_meta.parquet")
        meta["end_date"] = pd.to_datetime(meta["end_date"])
        valid_rows = meta["end_date"] <= cutoff

        if valid_rows.sum() < 500:
            return None, None  # too few usable rows

        X = pd.read_parquet(prefix + "_X.parquet")
        y = pd.read_parquet(prefix + "_y.parquet")

        X = X[valid_rows.values]
        y = y[valid_rows.values]

        if y.std().mean() < 1e-4:
            return None, None

        return X, y

    except Exception as e:
        print(f"[{ticker}] ⚠️ Load error: {e}")
        return None, None

def train_on_shards(window: int, model_type: str, out_dir: str, cutoff: str, save_every: int):
    cutoff_dt = pd.to_datetime(cutoff)
    tickers = get_tickers(out_dir, window)

    print(f"Training on shards up to {cutoff_dt.date()} with model={model_type} window={window}")

    # Choose model
    if model_type == "ridge":
        base_model = Ridge(alpha=1.0)
    elif model_type == "xgb":
        base_model = XGBRegressor(
            tree_method="hist",  # formerly gpu_hist
            device="cuda",
            max_bin=64,
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            verbosity=1
        )
    else:
        raise ValueError("model_type must be 'ridge' or 'xgb'")

    model = MultiOutputRegressor(base_model)

    trained_count = 0
    for ticker in tqdm(tickers, desc="Training shards"):
        X, y = load_and_filter_shard(ticker, window, out_dir, cutoff_dt)
        if X is None or y is None:
            continue

        # Clean bad values for GPU compatibility
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

        model.fit(X, y)
        trained_count += 1

        with open("trained_shards_cutoff.txt", "a") as f:
            f.write(f"{ticker}\n")

        if trained_count % save_every == 0:
            ckpt_path = f"mh_{model_type}_{window}_cutoff.pkl"
            joblib.dump(model, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    final_path = f"mh_{model_type}_{window}_cutoff_final.pkl"
    joblib.dump(model, final_path)
    print(f"✅ Finished training. Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=126, help="Input window size")
    parser.add_argument("--model", choices=["ridge", "xgb"], default="xgb")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory with shard files")
    parser.add_argument("--cutoff", type=str, default="2022-12-31", help="Max end_date for training data")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N tickers")
    args = parser.parse_args()

    train_on_shards(
        window=args.window,
        model_type=args.model,
        out_dir=args.out_dir,
        cutoff=args.cutoff,
        save_every=args.save_every
    )
