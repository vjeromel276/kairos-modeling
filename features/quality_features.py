#!/usr/bin/env python3
"""
Generate quality and efficiency signals from SHARADAR SF1 fundamentals.
Includes SAFF (Smart Attenuated Forward Fill) via expand_and_saff.
"""

import duckdb
import pandas as pd
import numpy as np
import argparse
from features.features_utils import expand_and_saff

def compute_quality_features(con):
    # 1) pull raw SF1
    df_raw = con.execute("""
        SELECT
          ticker,
          datekey AS date,
          netinc    AS ni,
          equity,
          revenue,
          assets,
          liabilities,
          ebitda,
          depamor
        FROM sf1
        WHERE ticker IS NOT NULL
        ORDER BY ticker, date
    """).fetchdf()

    # 2) dedupe to ensure unique (ticker, date)
    df_raw = df_raw.drop_duplicates(subset=["ticker", "date"])

    # 3) define which columns need SAFF
    value_cols = ["ni", "equity", "revenue", "assets", "liabilities", "ebitda", "depamor"]

    # 4) expand to full ticker×date grid and SAFF-fill
    df = expand_and_saff(
        con=con,
        sparse_df=df_raw,
        base_table="sep_base_common",
        value_cols=value_cols,
        ticker_col="ticker",
        date_col="date",
    )

    # 5) filter out rows with still-missing or zero/invalid denominators
    df = df[
        df["ni"].notna() &
        (df["equity"] > 0) &
        (df["revenue"] > 0) &
        (df["assets"] > 0) &
        (df["liabilities"] > 0)
    ]

    # 6) compute quality metrics
    df["roe"]           = df["ni"] / df["equity"]
    df["gross_margin"]  = df["ebitda"] / df["revenue"]
    df["net_margin"]    = df["ni"] / df["revenue"]
    df["asset_turnover"]= df["revenue"] / df["assets"]
    df["cfo"]           = df["ebitda"] - df["depamor"]
    df["accruals"]      = (df["ni"] - df["cfo"]) / df["assets"]

    # drop infinities / any remaining nulls
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["roe","gross_margin","net_margin","asset_turnover","accruals"]
    )
    
    # 7) Winsorize the raw quality metrics at [1%,99%] **per date** so we keep TSLA, etc.
    from features.features_utils import winsorize
    z_cols = ["roe","gross_margin","net_margin","asset_turnover","accruals"]
    df = winsorize(
        df,
        cols=z_cols,
        lower_q=0.01,
        upper_q=0.99,
        by_date=True,
        date_col="date"
    )
    print("✅ Winsorized quality metrics at [1%,99%] per date.")

    # 8) per-date z-scores
    z_cols = ["roe","gross_margin","net_margin","asset_turnover","accruals"]
    def add_z_scores(g):
        for c in z_cols:
            mu, sigma = g[c].mean(), g[c].std()
            g[f"{c}_z"] = (g[c] - mu) / sigma if sigma else np.nan
        return g

    df = df.groupby("date", group_keys=False).apply(add_z_scores).dropna(
        subset=[f"{c}_z" for c in z_cols]
    ) # type: ignore

    # 8) select final columns
    cols = (
        ["ticker","date"]
        + z_cols
        + [f"{c}_z" for c in z_cols]
    )
    return df[cols]

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
