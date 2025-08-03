#!/usr/bin/env python3
"""
DEBUG version: Extract valuation features from SHARADAR DAILY table.
Includes SAFF (Smart Attenuated Forward Fill) logic.
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

    df = df.sort_values(["ticker", "date"])
    for col in value_cols:
        last_valid = df.groupby("ticker")[col].ffill()
        gap = df.groupby("ticker")[f"_isnull_{col}"].cumsum()
        weight = np.exp(-0.05 * gap)  # 0.05 decay constant (adjustable)
        df[col] = last_valid * weight + median_vals[col] * (1 - weight)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_isnull_")])
    return df

def compute_fundamental_features(con):
    print("📥 Loading raw SHARADAR daily fundamental data...")
    df = con.execute("""
        SELECT ticker, date, marketcap, ev, evebit, evebitda, pe, pb, ps
        FROM daily
        ORDER BY ticker, date
    """).fetchdf()

    print("✅ Initial load complete. Columns:", df.columns.tolist())

    value_cols = ["marketcap", "ev", "evebit", "evebitda", "pe", "pb", "ps"]
    df = apply_saff(df, value_cols)
    print("✅ SAFF fill complete.")

    # Filtering
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

    # Feature engineering
    df["log_marketcap"] = np.log(df["marketcap"])
    df["ev_to_ebitda"] = df["ev"] / df["evebitda"]
    df["ev_to_ebit"] = df["ev"] / df["evebit"]

    def safe_inverse(x): return 1 / x.replace(0, np.nan)
    df["value_composite"] = (
        safe_inverse(df["pe"]) +
        safe_inverse(df["pb"]) +
        safe_inverse(df["ps"])
    )

    z_cols = ["log_marketcap", "pe", "pb", "ps", "ev_to_ebitda", "ev_to_ebit", "value_composite"]

    print(f"🧪 Starting per-date z-score computation across {df['date'].nunique()} dates...")

    z_dfs = []
    for i, (date, group) in enumerate(df.groupby("date"), 1):
        g = group.copy()
        for col in z_cols:
            mu = g[col].mean()
            std = g[col].std()
            g[f"{col}_z"] = (g[col] - mu) / std if std else np.nan
        z_dfs.append(g)

    df = pd.concat(z_dfs, ignore_index=True)
    df = df.dropna()

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
