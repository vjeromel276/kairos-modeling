#!/usr/bin/env python3
"""
Simulate a top-K 5-day hold strategy using shard predictions.

- Ranks tickers by ret_5d_pred per rebalance date
- Buys top K
- Computes realized return using actual ret_5d_f from y shards
- Tracks Sharpe, accuracy, etc.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_all_predictions(shard_dir, window):
    pred_files = glob.glob(os.path.join(shard_dir, f"mh_{window}_*_pred.parquet"))
    all_dfs = []
    for file in tqdm(pred_files, desc="Loading predictions"):
        df = pd.read_parquet(file)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def load_ground_truth(shard_dir, window):
    y_files = glob.glob(os.path.join(shard_dir, f"mh_{window}_*_y.parquet"))
    meta_files = glob.glob(os.path.join(shard_dir, f"mh_{window}_*_meta.parquet"))
    data = []
    for y_file, meta_file in tqdm(zip(sorted(y_files), sorted(meta_files)), desc="Loading y truth", total=len(y_files)):
        try:
            y_df = pd.read_parquet(y_file)
            meta_df = pd.read_parquet(meta_file)
            ticker = os.path.basename(y_file).split("_")[2]
            joined = pd.concat([meta_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
            joined["ticker"] = ticker
            data.append(joined)
        except Exception as e:
            print(f"[{y_file}] ⚠️ {e}")
    return pd.concat(data, ignore_index=True)

def simulate_top_k(preds_df, truth_df, k=50):
    merged = preds_df.merge(truth_df, on=["ticker", "start_date", "end_date"])
    merged = merged.sort_values("end_date")

    results = []

    for date, group in merged.groupby("end_date"):
        topk = group.sort_values("ret_5d_pred", ascending=False).head(k)
        mean_pred = topk["ret_5d_pred"].mean()
        mean_actual = topk["ret_5d_f"].mean()
        accuracy = (np.sign(topk["ret_5d_pred"]) == np.sign(topk["ret_5d_f"])).mean()

        results.append({
            "date": date,
            "mean_pred": mean_pred,
            "mean_actual": mean_actual,
            "accuracy": accuracy
        })

    df_result = pd.DataFrame(results)
    # Convert log return to percent return before compounding
    df_result["actual_pct_return"] = np.exp(df_result["mean_actual"]) - 1
    df_result["cumulative_return"] = (1 + df_result["actual_pct_return"]).cumprod()
    df_result["rolling_sharpe"] = (
        df_result["mean_actual"].rolling(20).mean() /
        df_result["mean_actual"].rolling(20).std().replace(0, np.nan)
    )
    return df_result

def main(window, shard_dir, topk):
    preds = load_all_predictions(shard_dir, window)
    truth = load_ground_truth(shard_dir, window)
    result = simulate_top_k(preds, truth, k=topk)

    out_csv = os.path.join(shard_dir, f"strategy_sim_{window}_top{topk}.csv") 
    result.to_csv(out_csv, index=False)

    print(f"✅ Strategy simulation saved to: {out_csv}")
    print(f"📈 Final cumulative return: {result['cumulative_return'].iloc[-1]:.2f}")
    print(f"📊 Mean accuracy: {result['accuracy'].mean():.2%}")
    print(f"📊 Mean daily Sharpe (20-day rolling): {result['rolling_sharpe'].mean():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=126, help="Window size (126 or 252)")
    parser.add_argument("--shard-dir", type=str, required=True, help="Where prediction shards are stored")
    parser.add_argument("--topk", type=int, default=50, help="How many tickers to hold at each step")
    args = parser.parse_args()

    main(args.window, args.shard_dir, args.topk)
