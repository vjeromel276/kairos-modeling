#!/usr/bin/env python3
"""
Phase 9: train_mlm.py (Parquet version, multithreaded + GPU-safe)

Trains a GRU-based Market Language Model (MLM) on partitioned Parquet from feat_matrix_filled.
Predicts ret_5d_f from 100-day rolling windows.

Outputs: models/mlm_best.pt
"""

import argparse
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
from pathlib import Path

SEQUENCE_LENGTH = 100
TARGET_COL = 'ret_5d_f'
EXCLUDE_COLS = {
    'ticker', 'date',
    'ret_1d', 'ret_5d', 'ret_21d',
    'ret_1d_f', 'ret_5d_f', 'ret_21d_f',
    'alpha_mlm', 'alpha_composite_v33_regime'
}

class LazyParquetSequenceDataset(Dataset):
    def __init__(self, parquet_dir, stride=5):
        self.ds = ds.dataset(parquet_dir, format="parquet", partitioning="hive")
        self.ticker_paths = sorted([f for f in Path(parquet_dir).glob("ticker=*")])
        self.feature_cols = None
        self.index = []  # (ticker_path, i)
        self.ticker_frames = {}
        self.stride = stride

        for tp in tqdm(self.ticker_paths, desc="Indexing tickers"):
            df = self.ds.to_table(filter=(ds.field("ticker") == tp.name.split("=")[1])).to_pandas()
            df = df.sort_values("date").drop_duplicates(subset="date")
            df = df.ffill().bfill()
            if df[TARGET_COL].isna().all() or len(df) < SEQUENCE_LENGTH + 1:
                continue
            df = df.dropna().reset_index(drop=True)
            self.ticker_frames[tp.name.split("=")[1]] = df
            if self.feature_cols is None:
                self.feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
            for i in range(SEQUENCE_LENGTH, len(df), stride):
                self.index.append((tp.name.split("=")[1], i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker, i = self.index[idx]
        df = self.ticker_frames[ticker]
        X = df.iloc[i - SEQUENCE_LENGTH:i][self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std
        y = df.iloc[i][TARGET_COL]
        y = np.clip(y, -0.25, 0.25)
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        losses = []
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                if not torch.isfinite(loss):
                    continue
                val_losses.append(loss.item())

        avg_train = np.mean(losses) if losses else float('nan')
        avg_val = np.mean(val_losses) if val_losses else float('nan')
        print(f"Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

    return best_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to feat_matrix_filled_parquet directory")
    parser.add_argument("--output", default="models/mlm_best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    dataset = LazyParquetSequenceDataset(args.parquet)
    if len(dataset) == 0:
        print("❌ No valid samples found.")
        return

    sample_X, sample_y = dataset[0]
    print("Sample X mean:", sample_X.mean().item())
    print("Sample X std:", sample_X.std().item())
    print("Sample Y:", sample_y.item())

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRURegressor(input_dim=len(dataset.feature_cols)).to(device)

    print(f"Training on {device} with {len(dataset.feature_cols)} input features from Parquet...")
    best_state = train(model, train_loader, val_loader, device, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(best_state, args.output)
    print("✔ Model saved.")

if __name__ == "__main__":
    main()