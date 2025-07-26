#!/usr/bin/env python3
"""
Simulates a top-K trading strategy using model predictions.
Logs summary metrics to DuckDB table 'strategy_stats'.
"""

import pandas as pd
import argparse
import numpy as np
import duckdb
from datetime import datetime
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pred-file", required=True, help="Path to predictions CSV")
parser.add_argument("--k", type=int, default=50, help="Top-K tickers to hold each rebalance")
parser.add_argument("--hold", type=int, default=5, help="Holding period in days")
parser.add_argument("--tag", type=str, default="", help="Optional run label for tracking")
args = parser.parse_args()

print(f"📥 Loading predictions: {args.pred_file}")
df = pd.read_csv(args.pred_file)

# Required fields
if not {"ticker", "date", "ret_5d_f_pred", "ret_5d_f"}.issubset(df.columns):
    raise ValueError("Prediction file must include: ticker, date, ret_5d_f_pred, ret_5d_f.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "ticker"])

# Simulate top-K strategy
unique_dates = df["date"].sort_values().unique()
returns_by_day = []
turnover_by_day = []
prev_tickers = set()

for i, d in enumerate(unique_dates):
    day_df = df[df["date"] == d]
    top_df = day_df.sort_values("ret_5d_f_pred", ascending=False).head(args.k)
    avg_ret = top_df["ret_5d_f"].mean()
    returns_by_day.append(avg_ret)

    tickers_today = set(top_df["ticker"])
    if i > 0:
        turnover = 1 - len(tickers_today & prev_tickers) / args.k
        turnover_by_day.append(turnover)
    prev_tickers = tickers_today

# Summary stats
returns = np.array(returns_by_day)
sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / args.hold)
drawdown = (np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)).max()
turnover = np.mean(turnover_by_day)

print("\n📊 Strategy Performance Summary")
print(f"Top-K Size         : {args.k}")
print(f"Holding Period     : {args.hold} days")
print(f"Average 5d Return  : {returns.mean():.4f}")
print(f"Annualized Sharpe  : {sharpe:.2f}")
print(f"Max Drawdown       : {drawdown:.4f}")
print(f"Avg Turnover       : {turnover:.2f}")

# ------------------------
# Log to DuckDB
# ------------------------

# Parse model info from filename
filename = Path(args.pred_file).name
match = re.search(r"predictions_(\w+)_(\d{4})\.csv", filename)
if not match:
    raise ValueError("Filename must follow format: predictions_<model>_<year>.csv")

model, year = match.group(1), int(match.group(2))

# Save to DuckDB
con = duckdb.connect("data/kairos.duckdb")
con.execute("""
    CREATE TABLE IF NOT EXISTS strategy_stats (
        model TEXT,
        year INT,
        horizon TEXT,
        date TIMESTAMP,
        sharpe DOUBLE,
        avg_return DOUBLE,
        max_drawdown DOUBLE,
        avg_turnover DOUBLE,
        top_k INT,
        hold_days INT,
        run_tag TEXT
    )
""")

con.execute("""
    INSERT INTO strategy_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    model,
    year,
    "ret_5d_f",
    datetime.utcnow(),
    float(sharpe),
    float(returns.mean()),
    float(drawdown),
    float(turnover),
    args.k,
    args.hold,
    args.tag
))

print("✅ Logged to strategy_stats in DuckDB.")
