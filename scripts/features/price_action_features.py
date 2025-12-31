#!/usr/bin/env python3
"""
price_action_features.py
Uses optimized sep_base_academic instead of sep_base.
"""

import duckdb
import pandas as pd
import argparse

def compute_price_action_features(con):
    df = con.execute("""
        SELECT ticker, date, open, high, low, close
        FROM sep_base_academic
        ORDER BY ticker, date
    """).fetchdf()

    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
    df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret_21d"] = df.groupby("ticker")["close"].pct_change(21)

    df["hl_ratio"] = df["high"] / df["low"]
    df["co_ratio"] = df["close"] / df["open"]

    df["true_range"] = df["high"] - df["low"]
    df["range_pct"] = (df["high"] - df["low"]) / df["open"]

    df = df.dropna()

    return df[
        [
            "ticker", "date",
            "ret_1d", "ret_5d", "ret_21d",
            "hl_ratio", "co_ratio", "true_range", "range_pct"
        ]
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_price_action")

    df_feat = compute_price_action_features(con)
    con.execute("""
        CREATE TABLE feat_price_action AS 
        SELECT 
            ticker,
            CAST(date AS DATE) as date,
            ret_1d, ret_5d, ret_21d,
            hl_ratio, co_ratio, true_range, range_pct
        FROM df_feat
    """)

    print(f"✔ Saved {len(df_feat):,} rows to feat_price_action")

if __name__ == "__main__":
    main()
