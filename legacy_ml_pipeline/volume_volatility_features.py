"""
volume_volatility_features.py

Computes volume and volatility-based features from sep_base table in DuckDB.

Features:
- Volume z-score (21d)
- Dollar volume (close * volume)
- Volume % change from previous day
- Rolling return std dev (5d, 21d)
- Average True Range (ATR 14d)

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_volume_volatility

To run:
    python scripts/features/volume_volatility_features.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import argparse

def compute_volume_vol_features(con):
    df = con.execute("SELECT ticker, date, close, high, low, volume FROM sep_base ORDER BY ticker, date").fetchdf()

    # Dollar volume
    df["dollar_volume"] = df["close"] * df["volume"]

    # Volume z-score (21d rolling)
    df["vol_zscore_21d"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: (x - x.rolling(21).mean()) / x.rolling(21).std())
    )

    # Volume % change
    df["volume_pct_change_1d"] = df.groupby("ticker")["volume"].pct_change()

    # Daily return
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    # Rolling volatility
    df["ret_std_5d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(5).std())
    df["ret_std_21d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(21).std())

    # ATR (14d average true range)
    df["true_range"] = df["high"] - df["low"]
    df["atr_14d"] = df.groupby("ticker")["true_range"].transform(lambda x: x.rolling(14).mean())

    df = df.dropna()

    return df[[
        "ticker", "date",
        "dollar_volume", "vol_zscore_21d", "volume_pct_change_1d",
        "ret_std_5d", "ret_std_21d", "atr_14d"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_volume_volatility")

    df_feat = compute_volume_vol_features(con)
    con.execute("CREATE TABLE feat_volume_volatility AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_volume_volatility in {args.db}")

if __name__ == "__main__":
    main()
