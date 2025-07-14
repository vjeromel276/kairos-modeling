"""
statistical_features.py

Computes z-score and percentile-based indicators to assess price stretch,
statistical outliers, and mean reversion potential.

Features:
- Z-scores of close price and returns
- Return percentile rank vs rolling window
- Price distance from rolling max
- Simple mean-reversion flag

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_stat

To run:
    python scripts/features/statistical_features.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def compute_statistical_features(con):
    df = con.execute("SELECT ticker, date, close FROM sep_base ORDER BY ticker, date").fetchdf()
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    def compute_group_stats(group):
        group["close_mean_21d"] = group["close"].rolling(21).mean()
        group["close_std_21d"] = group["close"].rolling(21).std()
        group["ret_1d_std_21d"] = group["ret_1d"].rolling(21).std()
        group["rolling_max_21d"] = group["close"].rolling(21).max()
        return group

    df = df.groupby("ticker", group_keys=False).apply(compute_group_stats)

    df["close_zscore_21d"] = (df["close"] - df["close_mean_21d"]) / df["close_std_21d"]
    df["ret_1d_zscore_21d"] = df["ret_1d"] / df["ret_1d_std_21d"]

    def rolling_percentile_rank(x):
        temp = x.argsort()
        ranks = temp.argsort()
        return ranks[-1] / (len(x) - 1) if len(x) > 1 else 0.5

    df["ret_1d_rank_21d"] = (
        df.groupby("ticker")["ret_1d"]
        .transform(lambda x: x.rolling(21).apply(rolling_percentile_rank, raw=True))
    )

    df["price_pct_from_rolling_max_21d"] = (
        (df["close"] - df["rolling_max_21d"]) / df["rolling_max_21d"]
    )

    df["mean_reversion_flag"] = (df["close_zscore_21d"] < -2.0).astype(int)

    df = df.dropna()

    return df[[
        "ticker", "date",
        "close_zscore_21d", "ret_1d_zscore_21d", "ret_1d_rank_21d",
        "price_pct_from_rolling_max_21d", "mean_reversion_flag"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_stat")

    df_stat = compute_statistical_features(con)
    con.execute("CREATE TABLE feat_stat AS SELECT * FROM df_stat")
    print(f"✅ Saved {len(df_stat):,} statistical features to 'feat_stat' table in {args.db}")

if __name__ == "__main__":
    main()
