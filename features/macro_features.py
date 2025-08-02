#!/usr/bin/env python3
"""
Build macro & market regime features from indicators table and auto-expand
across tickers for matrix joining.

Final output:
    feat_macro(ticker, date, [macro features...])
"""

import duckdb
import pandas as pd
import argparse
import numpy as np

def compute_macro_features(con):
    # Load indicators
    df = con.execute("""
        SELECT * FROM indicators
        WHERE ticker IN ('VIX', 'SPY', 'FEDFUNDS', 'DGS10')
        ORDER BY date
    """).fetchdf()

    df["value"] = df["value"].astype(float)

    # Pivot by date
    df = df.pivot(index="date", columns="ticker", values="value").sort_index()
    df = df.dropna()

    # Feature engineering
    df["spy_5d_return"] = df["SPY"].pct_change(5)
    df["vix_change_5d"] = df["VIX"].diff(5)
    df["term_spread"] = df["DGS10"] - df["FEDFUNDS"]
    df["regime_high_vol"] = (df["VIX"] > 25).astype(int)

    df = df.dropna().reset_index()

    return df

def expand_macro_to_tickers(con, macro_df):
    # Load all tickers from sep_base
    tickers = con.execute("SELECT DISTINCT ticker FROM sep_base").fetchdf()
    tickers["key"] = 1
    macro_df["key"] = 1

    # Cross-join on key to expand by ticker
    expanded = pd.merge(tickers, macro_df, on="key").drop(columns="key")
    return expanded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_macro")

    macro_df = compute_macro_features(con)
    expanded_df = expand_macro_to_tickers(con, macro_df)

    con.register("expanded_df", expanded_df)
    con.execute("CREATE TABLE feat_macro AS SELECT * FROM expanded_df")

    print(f"✅ Saved {len(expanded_df):,} macro-expanded rows to feat_macro")

if __name__ == "__main__":
    main()
