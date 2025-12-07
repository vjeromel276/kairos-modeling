#!/usr/bin/env python3
"""
Phase 9 Inference Script: predict_mlm.py

Loads a trained MLM (PyTorch) model and generates `alpha_mlm` for each (ticker, date) in the feature matrix.
Writes output as DuckDB table `feat_mlm` (ticker, date, alpha_mlm).

Assumes:
- feat_matrix contains 100+ days of history per ticker
- Model was trained on fixed-length windows (e.g. 100 days)
- Data is clean and sorted by ticker/date

Usage:
    python predict_mlm.py --model models/mlm_best.pt --parquet scripts/feature_matrices/feat_matrix_filled_parquet --db data/kairos.duckdb --table feat_mlm --output predictions/mlm_predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import pyarrow.dataset as ds
import duckdb
import os

SEQUENCE_LENGTH = 100
EXCLUDE_COLS = {
    'ticker', 'date',
    'ret_1d', 'ret_5d', 'ret_21d',
    'ret_1d_f', 'ret_5d_f', 'ret_21d_f',
    'alpha_mlm', 'alpha_composite_v33_regime'
}

class LazyParquetPredictionDataset(Dataset):
    def __init__(self, parquet_dir, window=100, stride=5):
        self.ds = ds.dataset(parquet_dir, format="parquet", partitioning="hive")
        self.ticker_paths = sorted([f for f in Path(parquet_dir).glob("ticker=*")])
        self.feature_cols = None
        self.index = []
        self.frames = {}
        self.window = window
        self.stride = stride

        for tp in tqdm(self.ticker_paths, desc="Indexing tickers"):
            ticker = tp.name.split("=")[1]
            df = self.ds.to_table(filter=(ds.field("ticker") == ticker)).to_pandas()
            df = df.sort_values("date").drop_duplicates(subset="date")
            df = df.ffill().bfill()
            if len(df) < self.window + 1:
                continue
            df = df.dropna().reset_index(drop=True)
            self.frames[ticker] = df
            if self.feature_cols is None:
                self.feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
            for i in range(self.window, len(df), self.stride):
                self.index.append((ticker, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker, i = self.index[idx]
        df = self.frames[ticker]
        X = df.iloc[i - self.window:i][self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std
        date = df.iloc[i]["date"]
        return torch.tensor(X, dtype=torch.float32), ticker, str(date)

class GRURegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)

def run_prediction(model, dataloader, device):
    output = []
    model.eval()
    with torch.no_grad():
        for X_batch, tickers, dates in tqdm(dataloader, desc="Scoring sequences"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            output.extend([(t, d, float(a)) for t, d, a in zip(tickers, dates, preds)])
    return pd.DataFrame(output, columns=["ticker", "date", "alpha_mlm"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="feat_mlm")
    parser.add_argument("--output", default="predictions/mlm_predictions.csv")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Parquet features...")
    dataset = LazyParquetPredictionDataset(args.parquet, window=args.window)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2
    )

    print("Rebuilding model and loading weights...")
    sample_X, _, _ = dataset[0]
    model = GRURegressor(input_dim=sample_X.shape[1]).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    df_out = run_prediction(model, dataloader, device)

    print("Writing to DuckDB and CSV...")
    con = duckdb.connect(args.db)
    con.execute(f"DROP TABLE IF EXISTS {args.table}")
    con.register("df_out", df_out)
    con.execute(f"CREATE TABLE {args.table} AS SELECT * FROM df_out")
    con.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"✔ Saved {len(df_out):,} predictions to {args.table} and {args.output}")

if __name__ == "__main__":
    main()
