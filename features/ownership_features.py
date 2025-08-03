#!/usr/bin/env python3
"""
Build daily ownership-based features from institutional data (sf3a + sf3b),  
using SAFF to attenuate the forward fill between quarter‐ends.
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

def apply_saff(df, value_cols, decay=0.05):
    """
    Smart Attenuated Forward Fill:
      - ffill() each series within ticker
      - compute gap = consecutive nulls
      - weight = exp(-decay * gap)
      - interpolate toward median when gap > 0
    """
    med = df.groupby("ticker")[value_cols].transform("median")
    # mark nulls
    for c in value_cols:
        df[f"_null_{c}"] = df[c].isna().astype(int)
    df = df.sort_values(["ticker","date"])
    # ffill + decay
    for c in value_cols:
        last = df.groupby("ticker")[c].ffill()
        gap  = df.groupby("ticker")[f"_null_{c}"].cumsum()
        w    = np.exp(-decay * gap)
        df[c] = last * w + med[c] * (1 - w)
    # cleanup
    df = df.drop(columns=[col for col in df if col.startswith("_null_")])
    return df

def compute_ownership_features(con):
    # 1) build full daily grid
    calendar = con.execute("""
      SELECT ticker, date
      FROM sep_base
      WHERE ticker IS NOT NULL
    """).fetchdf()

    # 2) raw SF3A inst data
    raw = con.execute("""
      SELECT
        calendardate::date AS date,
        ticker,
        shrvalue AS inst_value,
        shrunits AS inst_units,
        shrholders AS inst_holders
      FROM sf3a
      WHERE ticker IS NOT NULL
    """).fetchdf()

    # 3) left-join onto every trading day
    inst = pd.merge(
      calendar,
      raw,
      on=["ticker","date"],
      how="left"
    )

    # 4) SAFF imputation on the three columns
    inst = apply_saff(inst, ["inst_value","inst_units","inst_holders"])

    # 5) recompute quarter-to-quarter changes (now daily but mostly zero except at quarter turns)
    inst = inst.sort_values(["ticker","date"])
    inst["inst_value_change"]  = inst.groupby("ticker")["inst_value"].pct_change()
    inst["inst_units_change"]  = inst.groupby("ticker")["inst_units"].pct_change()
    inst["inst_holder_change"] = inst.groupby("ticker")["inst_holders"].diff()

    # 6) optional: drop first row per ticker (no meaningful change)
    inst = inst.dropna(subset=["inst_value_change","inst_units_change","inst_holder_change"])

    # 7) return only the feature columns
    return inst[[
      "ticker","date",
      "inst_value","inst_units","inst_holders",
      "inst_value_change","inst_units_change","inst_holder_change"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_ownership")

    df = compute_ownership_features(con)
    con.execute("CREATE TABLE feat_ownership AS SELECT * FROM df")
    print(f"✅ Saved {len(df):,} rows to feat_ownership in {args.db}")

if __name__ == "__main__":
    main()
