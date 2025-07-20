#!/usr/bin/env python3
"""
Train a multi-output regression model from memory-efficient per-ticker shards.

Each shard group:
    mh_<window>_<TICKER>_X.parquet
    mh_<window>_<TICKER>_y.parquet
    mh_<window>_<TICKER>_meta.parquet

Supports:
- LightGBM or Ridge
- Resume from last ticker
- Save every N tickers

Usage:
    python train_model_shards.py --window 252 --model lgbm --resume-from AAPL --save-every 10
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
from lightgbm import LGBMRegressor


def get_tickers(out_dir: str, window: int):
    x_files = glob.glob(os.path.join(out_dir, f"mh_{window}_*_X.parquet"))
    tickers = sorted([os.path.basename(f).split("_")[2] for f in x_files])
    return tickers


def load_shard(ticker: str, window: int, out_dir: str):
    base = os.path.join(out_dir, f"mh_{window}_{ticker}")
    X = pd.read_parquet(base + "_X.parquet")
    y = pd.read_parquet(base + "_y.parquet")
    return X, y


def train_on_shards(window: int, model_type: str, out_dir: str, resume_from: str = None, save_every: int = 10): # type: ignore
    tickers = get_tickers(out_dir, window)

    if resume_from:
        tickers = [t for t in tickers if t >= resume_from]

    print(f"Training on {len(tickers)} tickers, window={window}, model={model_type}")

    # Choose base model
    if model_type == "ridge":
        base_model = Ridge(alpha=1.0)
    elif model_type == "lgbm":
        base_model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    else:
        raise ValueError("model_type must be 'ridge' or 'lgbm'")

    model = MultiOutputRegressor(base_model) # type: ignore

    trained_count = 0
    for ticker in tqdm(tickers, desc="Training shards"):
        try:
            X, y = load_shard(ticker, window, out_dir)

            if len(X) < 500:
                print(f"[{ticker}] Skipped: only {len(X)} samples")
                continue
            if y.std().mean() < 1e-4:
                print(f"[{ticker}] Skipped: target too flat (std={y.std().mean():.6f})")
                continue

            model.partial_fit(X, y) if hasattr(model, "partial_fit") else model.fit(X, y)
            trained_count += 1

            with open("trained_shards.txt", "a") as f:
                f.write(f"{ticker}\n")

            if trained_count % save_every == 0:
                out_path = f"mh_{model_type}_{window}_ckpt.pkl"
                joblib.dump(model, out_path)
                print(f"💾 Saved checkpoint: {out_path}")

        except Exception as e:
            print(f"[{ticker}] ⚠️ Error: {e}")
            continue

    final_model_path = f"mh_{model_type}_{window}.pkl"
    joblib.dump(model, final_model_path)
    print(f"✅ Finished training. Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=252, help="Window size (126 or 252)")
    parser.add_argument("--model", type=str, choices=["ridge", "lgbm"], default="lgbm")
    parser.add_argument("--out-dir", type=str, default="data/shards/", help="Directory with shard files")
    parser.add_argument("--resume-from", type=str, help="Resume from this ticker onward")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N tickers")
    args = parser.parse_args()

    train_on_shards(
        window=args.window,
        model_type=args.model,
        out_dir=args.out_dir,
        resume_from=args.resume_from,
        save_every=args.save_every
    )
# '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/'