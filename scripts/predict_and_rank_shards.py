#!/usr/bin/env python3
"""
predict_and_rank_shards.py

Streams through X shards, loads trained model,
predicts y_hat per ticker, joins with meta,
and writes per-ticker predictions to:
  mh_<window>_<TICKER>_pred.parquet
"""

import argparse
import os
import joblib
import pandas as pd
import glob
from tqdm import tqdm

def main(window: int, model_path: str, shard_dir: str):
    model = joblib.load(model_path)
    x_files = sorted(glob.glob(os.path.join(shard_dir, f"mh_{window}_*_X.parquet")))

    for x_path in tqdm(x_files, desc="Predicting shards"):
        ticker = os.path.basename(x_path).split("_")[2]
        out_path = os.path.join(shard_dir, f"mh_{window}_{ticker}_pred.parquet")
        if os.path.exists(out_path):
            continue  # already predicted

        try:
            X = pd.read_parquet(x_path)
            meta_path = os.path.join(shard_dir, f"mh_{window}_{ticker}_meta.parquet")
            meta_df = pd.read_parquet(meta_path)

            y_pred = model.predict(X)
            df_pred = pd.DataFrame(y_pred, columns=["ret_1d_pred", "ret_5d_pred", "ret_21d_pred"])
            df_pred["ticker"] = ticker

            # Align meta + predictions
            df_out = pd.concat([meta_df.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
            df_out.to_parquet(out_path, index=False)

        except Exception as e:
            print(f"[{ticker}] ⚠️ Prediction failed: {e}")

    print("✅ All predictions written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True, help="Input window size (e.g. 126 or 252)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (e.g. mh_lgbm_126.pkl)")
    parser.add_argument("--shard-dir", type=str, default="data/shards/", help="Directory with input shards")
    args = parser.parse_args()
    main(args.window, args.model, args.shard_dir)
