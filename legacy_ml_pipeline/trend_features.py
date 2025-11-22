# scripts/features/trend_features.py

"""
trend_features.py

Generates trend-based features using moving averages and momentum indicators.
Focuses on smoothing and slope dynamics over short and medium horizons.

Features:
- Simple and exponential moving averages (5, 12, 21, 26 days)
- Price position vs SMA
- Slope of SMA (5-day difference)
- MACD, MACD signal, MACD histogram

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_trend

To run:
    python scripts/features/trend_features.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def compute_trend_features(con):
    df = con.execute("SELECT ticker, date, close FROM sep_base ORDER BY ticker, date").fetchdf()

    df["sma_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
    df["sma_21"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(21).mean())
    df["ema_12"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["ema_26"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["price_vs_sma_21"] = (df["close"] - df["sma_21"]) / df["sma_21"]
    df["sma_21_slope"] = df.groupby("ticker")["sma_21"].transform(lambda x: x - x.shift(5))

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df = df.dropna()
    return df[[
        "ticker", "date", "sma_5", "sma_21", "ema_12", "ema_26",
        "price_vs_sma_21", "sma_21_slope", "macd", "macd_signal", "macd_hist"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_trend")

    df_trend = compute_trend_features(con)
    con.execute("CREATE TABLE feat_trend AS SELECT * FROM df_trend")
    print(f"✅ Saved {len(df_trend):,} trend features to 'feat_trend' table in {args.db}")

if __name__ == "__main__":
    main()
