#!/usr/bin/env python3
"""
DEBUG version: Extract valuation features from SHARADAR DAILY table.

Includes detailed logs to trace column loss (esp. 'ticker').
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def compute_fundamental_features(con):
    print("📥 Loading raw SHARADAR daily fundamental data...")
    df = con.execute("""
        SELECT ticker, date, marketcap, ev, evebit, evebitda, pe, pb, ps
        FROM daily
        ORDER BY ticker, date
    """).fetchdf()

    print("✅ Initial load complete. Columns:", df.columns.tolist())

    # Forward-fill per ticker
    df = df.set_index(["ticker", "date"]).groupby("ticker").ffill().reset_index()
    print("✅ Forward-fill complete. Columns now:", df.columns.tolist())

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
    print("Columns post-filtering:", df.columns.tolist())

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

    # Verify required columns
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("❌ 'ticker' or 'date' missing before z-scoring!")

    print(f"🧪 Starting per-date z-score computation across {df['date'].nunique()} dates...")

    z_dfs = []
    for i, (date, group) in enumerate(df.groupby("date"), 1):
        if "ticker" not in group.columns:
            print(f"❌ Group on {date} missing 'ticker'. Skipping.")
            continue

        g = group.copy()
        for col in z_cols:
            mu = g[col].mean()
            std = g[col].std()
            g[f"{col}_z"] = (g[col] - mu) / std if std else np.nan

        if i % 250 == 0 or i == 1:
            print(f"  ✔ Processed date: {date} | rows: {len(g)} | cols: {g.columns.tolist()}")
        z_dfs.append(g)

    df = pd.concat(z_dfs, ignore_index=True)
    print("✅ Z-scoring complete. Final frame shape:", df.shape)
    print("Final columns:", df.columns.tolist())

    if "ticker" not in df.columns:
        raise ValueError("🔥 After concat, 'ticker' is missing!")

    df = df.dropna()
    print(f"✅ Final dropna() complete. {len(df):,} rows ready.")

    # Output columns
    cols = ["ticker", "date"] + z_cols + [f"{col}_z" for col in z_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"❌ Missing expected columns: {missing}")

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
