#!/usr/bin/env python3
"""
predict_and_rank_cutoff.py

Streams through X shards, loads trained model,
predicts y_hat per ticker, filters to post-cutoff dates,
and writes per-ticker predictions to:
  mh_<window>_<TICKER>_pred.parquet

Usage:
    python predict_and_rank_cutoff.py \
        --window 126 \
        --model mh_lgbm_126_cutoff_final.pkl \
        --shard-dir /path/to/shards \
        --cutoff 2023-01-01
"""

import argparse
import os
import glob
import pandas as pd
import joblib
from tqdm import tqdm

def main(window: int, model_path: str, shard_dir: str, cutoff: str):
    model = joblib.load(model_path)
    x_files = sorted(glob.glob(os.path.join(shard_dir, f"mh_{window}_*_X.parquet")))
    cutoff_dt = pd.to_datetime(cutoff)

    for x_path in tqdm(x_files, desc="Predicting shards"):
        ticker = os.path.basename(x_path).split("_")[2]
        out_path = os.path.join(shard_dir, f"mh_{window}_{ticker}_pred.parquet")
        if os.path.exists(out_path):
            continue  # already predicted

        try:
            X = pd.read_parquet(x_path)
            meta_path = os.path.join(shard_dir, f"mh_{window}_{ticker}_meta.parquet")
            meta_df = pd.read_parquet(meta_path)
            meta_df["end_date"] = pd.to_datetime(meta_df["end_date"])

            y_pred = model.predict(X)
            df_pred = pd.DataFrame(y_pred, columns=["ret_1d_pred", "ret_5d_pred", "ret_21d_pred"])

            df_out = pd.concat([meta_df.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
            df_out = df_out[df_out["end_date"] >= cutoff_dt]  # keep only 2023+ rows

            if not df_out.empty:
                df_out.to_parquet(out_path, index=False)

        except Exception as e:
            print(f"[{ticker}] ⚠️ Prediction failed: {e}")

    print("✅ All post-cutoff predictions written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True, help="Input window size (e.g. 126 or 252)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--shard-dir", type=str, default="data/shards/", help="Directory with shard files")
    parser.add_argument("--cutoff", type=str, default="2023-01-01", help="Minimum end_date to keep")
    args = parser.parse_args()

    main(args.window, args.model, args.shard_dir, args.cutoff)
