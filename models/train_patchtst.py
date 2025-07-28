#!/usr/bin/env python3
"""
Train PatchTST on rolling [252, F] frames from feat_matrix_targets_2008.
This version limits to the first 50 tickers for test purposes.
"""

import duckdb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from datetime import datetime
import joblib
import os

# ------------------------
# PatchTST Encoder Module
# ------------------------

class PatchTST(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.embed(x)               # [B, T, F] → [B, T, d_model]
        x = self.encoder(x)             # [B, T, d_model]
        x = self.head(x.transpose(1, 2))# [B, T, d_model] → [B, d_model] → [B, 1]
        return x.squeeze(1)

# ------------------------
# DuckDB IterableDataset
# ------------------------

class DuckDBPatchTSTDataset(IterableDataset):
    def __init__(self, db_path, table, tickers, window=252, target_col='ret_5d_f'):
        self.db_path = db_path
        self.table = table
        self.tickers = tickers
        self.window = window
        self.target_col = target_col

    def __iter__(self):
        con = duckdb.connect(self.db_path)
        for ticker in self.tickers:
            df = con.execute(f"""
                SELECT * FROM {self.table}
                WHERE ticker = '{ticker}'
                ORDER BY date
            """).fetchdf()

            df = df.dropna()
            if df.shape[0] <= self.window:
                continue

            X = df.drop(columns=['ticker', 'date', 'ret_1d_f', 'ret_5d_f', 'ret_21d_f'])
            X = X.astype(np.float32).values

            y = df[self.target_col].values

            for i in range(self.window, len(df)):
                x_window = X[i-self.window:i]
                y_target = y[i]
                yield torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# ------------------------
# Training Setup
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

DB_PATH = "data/kairos.duckdb"
TABLE_NAME = "feat_matrix_targets_2008"
BATCH_SIZE = 64
EPOCHS = 5
OUTPUT_PATH = "models/output"

# Get 50 tickers
con = duckdb.connect(DB_PATH)
tickers = con.execute(f"""
    SELECT DISTINCT ticker FROM {TABLE_NAME}
    ORDER BY ticker LIMIT 50
""").fetchdf()["ticker"].tolist()

# Build dataset and dataloader
dataset = DuckDBPatchTSTDataset(DB_PATH, TABLE_NAME, tickers)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Model
num_features = con.execute(f"PRAGMA table_info('{TABLE_NAME}')").fetchdf()
num_features = len([c for c in num_features["name"].tolist()
                    if c not in ("ticker", "date", "ret_1d_f", "ret_5d_f", "ret_21d_f")])

model = PatchTST(input_dim=num_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ------------------------
# Training Loop
# ------------------------

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {np.mean(losses):.6f}")

# ------------------------
# Save Model
# ------------------------

os.makedirs(OUTPUT_PATH, exist_ok=True)
model_file = os.path.join(OUTPUT_PATH, f"patchtst_ret_5d_f_2008.pt")
torch.save(model.state_dict(), model_file)
print(f"✅ Model saved to {model_file}")
