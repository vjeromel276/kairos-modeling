#!/usr/bin/env python3
"""
volume_volatility_features.py — patched for sep_base_academic
"""

import duckdb
import pandas as pd
import argparse


def compute_volume_vol_features(con):
    df = con.execute(
        """
        SELECT ticker, date, close, high, low, volume
        FROM sep_base_academic
        ORDER BY ticker, date
    """
    ).fetchdf()

    df["dollar_volume"] = df["close"] * df["volume"]

    df["vol_zscore_21d"] = df.groupby("ticker")["volume"].transform(
        lambda x: (x - x.rolling(21).mean()) / x.rolling(21).std()
    )

    df["volume_pct_change_1d"] = df.groupby("ticker")["volume"].pct_change()

    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
    df["ret_std_5d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(5).std()
    )
    df["ret_std_21d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(21).std()
    )

    df["true_range"] = df["high"] - df["low"]
    df["atr_14d"] = df.groupby("ticker")["true_range"].transform(
        lambda x: x.rolling(14).mean()
    )

    df = df.dropna()

    return df[
        [
            "ticker",
            "date",
            "dollar_volume",
            "vol_zscore_21d",
            "volume_pct_change_1d",
            "ret_std_5d",
            "ret_std_21d",
            "atr_14d",
        ]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_volume_volatility")

    df_feat = compute_volume_vol_features(con)
    con.execute(
        """
                CREATE TABLE feat_volume_volatility AS 
                    SELECT 
                        "ticker",CAST(date AS DATE) as date,
                        "dollar_volume","vol_zscore_21d","volume_pct_change_1d",
                        "ret_std_5d","ret_std_21d","atr_14d"
                    FROM df_feat
    """
    )

    print(f"✔ Saved {len(df_feat):,} rows to feat_volume_volatility")


if __name__ == "__main__":
    main()
