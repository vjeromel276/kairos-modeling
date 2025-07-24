import pandas as pd
import glob
import os

shard_dir = "/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/scripts/shards"  # e.g., '/home/vjerome2/SSD_Quant/.../shards'
window = 126

# Collect all meta files used during training
meta_files = glob.glob(os.path.join(shard_dir, f"mh_{window}_*_meta.parquet"))

train_meta = []
for path in meta_files:
    ticker = os.path.basename(path).split("_")[2]
    df = pd.read_parquet(path)
    df["ticker"] = ticker
    train_meta.append(df[["ticker", "start_date", "end_date"]])

train_df = pd.concat(train_meta, ignore_index=True)
train_df["end_date"] = pd.to_datetime(train_df["end_date"])
train_df["start_date"] = pd.to_datetime(train_df["start_date"])

train_df.to_csv("train_window_metadata.csv", index=False)
