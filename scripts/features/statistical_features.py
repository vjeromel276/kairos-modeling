#!/usr/bin/env python3
"""
statistical_features.py — patched for sep_base_academic
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def compute_statistical_features(con):
    df = con.execute("""
        SELECT ticker, date, close
        FROM sep_base_academic
        ORDER BY ticker, date
    """).fetchdf()

    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    def grp_fn(g):
        g["close_mean_21d"] = g["close"].rolling(21).mean()
        g["close_std_21d"] = g["close"].rolling(21).std()
        g["ret_1d_std_21d"] = g["ret_1d"].rolling(21).std()
        g["rolling_max_21d"] = g["close"].rolling(21).max()
        return g

    df = df.groupby("ticker", group_keys=False).apply(grp_fn)

    df["close_zscore_21d"] = (df["close"] - df["close_mean_21d"]) / df["close_std_21d"]
    df["ret_1d_zscore_21d"] = df["ret_1d"] / df["ret_1d_std_21d"]

    df["mean_reversion_flag"] = (df["close_zscore_21d"] < -2).astype(int)

    df = df.dropna()

    return df[
        [
            "ticker","date",
            "close_zscore_21d","ret_1d_zscore_21d",
            "rolling_max_21d","mean_reversion_flag"
        ]
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_stat")

    df_feat = compute_statistical_features(con)
    con.execute("""
                CREATE TABLE feat_stat AS 
                    SELECT 
                        "ticker",CAST(date AS DATE) as date,
                        "close_zscore_21d","ret_1d_zscore_21d",
                        "rolling_max_21d","mean_reversion_flag"
                FROM df_feat""")

    print(f"✔ Saved {len(df_feat):,} rows to feat_stat")

if __name__ == "__main__":
    main()
