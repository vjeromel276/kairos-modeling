#!/usr/bin/env python3
"""
DEBUG version: Extract valuation features from SHARADAR DAILY table.
Includes SAFF (Smart Attenuated Forward Fill) logic applied over the full sep_base date range.
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def apply_saff(df, value_cols):
    print("🧠 Applying SAFF to impute missing fundamentals with decay...")
    median_vals = df.groupby("ticker")[value_cols].transform("median")
    for col in value_cols:
        df[f"_isnull_{col}"] = df[col].isna().astype(int)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in value_cols:
        last_valid = df.groupby("ticker")[col].ffill()
        gap = df.groupby("ticker")[f"_isnull_{col}"].cumsum()
        weight = np.exp(-0.05 * gap)  # 0.05 decay constant
        df[col] = last_valid * weight + median_vals[col] * (1 - weight)

    # cleanup
    df = df.drop(columns=[c for c in df.columns if c.startswith("_isnull_")])
    return df


def compute_fundamental_features(con):
    # 1) Load raw SHARADAR fundamentals
    print("📥 Loading raw SHARADAR daily fundamental data...")
    df = con.execute(
        """
        SELECT ticker, date, marketcap, ev, evebit, evebitda, pe, pb, ps
        FROM daily
        ORDER BY ticker, date
        """
    ).fetchdf()
    print("✅ Initial load complete. Got", len(df), "rows of fundamentals.")

    # 2) Get full trading dates from sep_base and extend each ticker
    print("🔍 Fetching full date range from sep_base...")
    dates = con.execute(
        "SELECT DISTINCT date FROM sep_base ORDER BY date"
    ).fetchdf()["date"]
    tickers = df["ticker"].unique()
    full_index = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    df = df.set_index(["ticker", "date"]).reindex(full_index).reset_index()
    print(f"🔄 Extended to full ticker-date index: {df.shape[0]:,} rows.")

    # 3) Apply SAFF on this complete frame
    value_cols = ["marketcap", "ev", "evebit", "evebitda", "pe", "pb", "ps"]
    df = apply_saff(df, value_cols)
    print("✅ SAFF fill complete.")

    # 4) Filter out extreme or invalid values
    df = df[
        (df["marketcap"] > 0) &
        (df["pe"] > 0) & (df["pe"] < 100) &
        (df["pb"] > 0) & (df["pb"] < 20) &
        (df["ps"] > 0) & (df["ps"] < 20) &
        (df["evebitda"] > 0) & (df["evebitda"] < 1e6) &
        (df["evebit"] > 0) & (df["evebit"] < 1e6) &
        (df["ev"] > 0)
    ]
    print(f"✅ Filtering complete: {len(df):,} rows remain.")

    # 5) Feature engineering
    df["log_marketcap"] = np.log(df["marketcap"])
    df["ev_to_ebitda"] = df["ev"] / df["evebitda"]
    df["ev_to_ebit"] = df["ev"] / df["evebit"]
    def safe_inverse(x): return 1 / x.replace(0, np.nan)
    df["value_composite"] = (
        safe_inverse(df["pe"]) +
        safe_inverse(df["pb"]) +
        safe_inverse(df["ps"] )
    )

    z_cols = ["log_marketcap", "pe", "pb", "ps", "ev_to_ebitda", "ev_to_ebit", "value_composite"]
    print(f"🧪 Computing per-date z-scores over {df['date'].nunique():,} dates...")

    z_dfs = []
    for date, group in df.groupby("date"):
        g = group.copy()
        for col in z_cols:
            mu = g[col].mean()
            std = g[col].std()
            g[f"{col}_z"] = (g[col] - mu) / std if std else np.nan
        z_dfs.append(g)
    df = pd.concat(z_dfs, ignore_index=True).dropna()

    # Select final columns
    cols = ["ticker", "date"] + z_cols + [f"{c}_z" for c in z_cols]
    return df[cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_fundamentals")

    df_feat = compute_fundamental_features(con)
    con.execute("CREATE TABLE feat_fundamentals AS SELECT * FROM df_feat")
    print(f"✅ Saved {len(df_feat):,} rows to feat_fundamentals in {args.db}")

if __name__ == "__main__":
    main()
