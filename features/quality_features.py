#!/usr/bin/env python3
"""
Generate quality and efficiency signals from SHARADAR SF1 fundamentals.
Includes SAFF (Smart Attenuated Forward Fill).
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
        weight = np.exp(-0.05 * gap)  # decay constant
        df[col] = last_valid * weight + median_vals[col] * (1 - weight)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_isnull_")])
    return df

def compute_quality_features(con):
    df = con.execute("""
        SELECT ticker, datekey AS date, netinc AS ni, equity, revenue, assets, liabilities, ebitda, depamor
        FROM sf1
        ORDER BY ticker, date
    """).fetchdf()

    value_cols = ["ni", "equity", "revenue", "assets", "liabilities", "ebitda", "depamor"]
    df = apply_saff(df, value_cols)

    # Filter invalid rows
    df = df[
        (df["ni"].notnull()) &
        (df["equity"] > 0) &
        (df["revenue"] > 0) &
        (df["assets"] > 0) &
        (df["liabilities"] > 0)
    ]

    # Quality metrics
    df["roe"] = df["ni"] / df["equity"]
    df["gross_margin"] = df["ebitda"] / df["revenue"]
    df["net_margin"] = df["ni"] / df["revenue"]
    df["asset_turnover"] = df["revenue"] / df["assets"]

    # Accruals proxy: (Net Income - CFO) / Assets
    df["cfo"] = df["ebitda"] - df["depamor"]
    df["accruals"] = (df["ni"] - df["cfo"]) / df["assets"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    def zscore(group, cols):
        for col in cols:
            mu = group[col].mean()
            std = group[col].std()
            group[f"{col}_z"] = (group[col] - mu) / std
        return group

    z_cols = ["roe", "gross_margin", "net_margin", "asset_turnover", "accruals"]
    df = df.groupby("date", group_keys=False).apply(lambda g: zscore(g, z_cols))

    return df[[
        "ticker", "date",
        "roe", "gross_margin", "net_margin", "asset_turnover", "accruals",
        "roe_z", "gross_margin_z", "net_margin_z", "asset_turnover_z", "accruals_z"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_quality")

    df_feat = compute_quality_features(con)
    con.execute("CREATE TABLE feat_quality AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_quality in {args.db}")

if __name__ == "__main__":
    main()
