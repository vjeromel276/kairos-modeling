"""
price_shape_features.py

Extracts candlestick shape and gap-related features from OHLCV data in DuckDB.

Features:
- Candle body, wick sizes
- Candle body and wick % of total range
- Gap up/down (absolute and %)

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_price_shape

To run:
    python scripts/features/price_shape_features.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import argparse

def compute_price_shape_features(con):
    df = con.execute("SELECT ticker, date, open, high, low, close FROM sep_base ORDER BY ticker, date").fetchdf()

    # Candlestick components
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]

    # Ratios
    df["body_pct_of_range"] = df["body_size"] / df["candle_range"]
    df["upper_wick_pct"] = df["upper_wick"] / df["candle_range"]
    df["lower_wick_pct"] = df["lower_wick"] / df["candle_range"]

    # Gap features
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["gap_open"] = df["open"] - df["prev_close"]
    df["gap_pct"] = df["gap_open"] / df["prev_close"]

    df = df.dropna()

    return df[[
        "ticker", "date",
        "body_size", "upper_wick", "lower_wick", "candle_range",
        "body_pct_of_range", "upper_wick_pct", "lower_wick_pct",
        "gap_open", "gap_pct"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_price_shape")

    df_feat = compute_price_shape_features(con)
    con.execute("CREATE TABLE feat_price_shape AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_price_shape table in {args.db}")

if __name__ == "__main__":
    main()
