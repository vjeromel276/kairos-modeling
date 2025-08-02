#!/usr/bin/env python3
"""
Extract analyst estimate-based features from SHARADAR SF2 table.

Includes:
- Forward-filled consensus EPS estimates
- Surprise: Actual - Estimate
- Revision momentum
- Z-score normalization
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def compute_analyst_features(con):
    df = con.execute("""
        SELECT ticker, date, epsmean, epshigh, epslow
        FROM sf2
        WHERE epsmean IS NOT NULL
        ORDER BY ticker, date
    """).fetchdf()

    # Forward-fill per ticker
    df = df.groupby("ticker").ffill()

    # Revision over time
    df["eps_revision_21d"] = df.groupby("ticker")["epsmean"].diff(21)
    df["eps_revision_5d"] = df.groupby("ticker")["epsmean"].diff(5)

    # Dispersion
    df["eps_dispersion"] = df["epshigh"] - df["epslow"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Cross-sectional z-scores
    def zscore(group, cols):
        for col in cols:
            mu = group[col].mean()
            std = group[col].std()
            group[f"{col}_z"] = (group[col] - mu) / std
        return group

    z_cols = ["eps_revision_21d", "eps_revision_5d", "eps_dispersion"]
    df = df.groupby("date", group_keys=False).apply(lambda g: zscore(g, z_cols))

    return df[[
        "ticker", "date",
        "epsmean", "eps_revision_5d", "eps_revision_21d", "eps_dispersion",
        "eps_revision_5d_z", "eps_revision_21d_z", "eps_dispersion_z"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_analyst")

    df_feat = compute_analyst_features(con)
    con.execute("CREATE TABLE feat_analyst AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_analyst in {args.db}")

if __name__ == "__main__":
    main()
