"""
price_action_features.py

Generates price action and momentum-based features from the base SEP OHLCV dataset.

Features:
- Daily returns
- Rolling N-day returns (e.g., 5d, 21d)
- Price ratios (high/low, close/open)
- True range and price range percentage

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_price_action

To run:
    python scripts/features/price_action_features.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import argparse

def compute_price_action_features(con):
    df = con.execute("SELECT ticker, date, open, high, low, close FROM sep_base ORDER BY ticker, date").fetchdf()

    # Compute returns
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
    df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret_21d"] = df.groupby("ticker")["close"].pct_change(21)

    # Price ratios
    df["hl_ratio"] = df["high"] / df["low"]
    df["co_ratio"] = df["close"] / df["open"]

    # True range and range percentage
    df["true_range"] = df["high"] - df["low"]
    df["range_pct"] = (df["high"] - df["low"]) / df["open"]

    df = df.dropna()

    return df[[
        "ticker", "date",
        "ret_1d", "ret_5d", "ret_21d",
        "hl_ratio", "co_ratio", "true_range", "range_pct"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_price_action")

    df_feat = compute_price_action_features(con)
    con.execute("CREATE TABLE feat_price_action AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_price_action table in {args.db}")

if __name__ == "__main__":
    main()
