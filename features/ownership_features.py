#!/usr/bin/env python3
"""
Build ownership-based features from institutional data (sf3a + sf3b)

Features:
- institutional_ownership_pct
- institutional_churn_1q
- institutional_holder_count
- institutional_value_held
"""

import duckdb
import pandas as pd
import argparse

def compute_ownership_features(con):
    # Load sf3a: institutional holdings by ticker per quarter
    inst = con.execute("""
        SELECT
            calendardate AS date,
            ticker,
            shrvalue AS inst_value,
            shrunits AS inst_units,
            shrholders AS inst_holders
        FROM sf3a
        WHERE ticker IS NOT NULL AND shrvalue IS NOT NULL AND shrunits IS NOT NULL
        ORDER BY ticker, date
    """).fetchdf()

    # Compute percent ownership and changes
    inst = inst.sort_values(["ticker", "date"])
    inst["inst_value_change"] = inst.groupby("ticker")["inst_value"].pct_change()
    inst["inst_units_change"] = inst.groupby("ticker")["inst_units"].pct_change()
    inst["inst_holder_change"] = inst.groupby("ticker")["inst_holders"].diff()

    # Drop early rows with missing diffs
    inst = inst.dropna(subset=["inst_value_change", "inst_units_change", "inst_holder_change"])

    return inst[[
        "ticker", "date",
        "inst_value", "inst_units", "inst_holders",
        "inst_value_change", "inst_units_change", "inst_holder_change"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_ownership")

    df_feat = compute_ownership_features(con)
    con.execute("CREATE TABLE feat_ownership AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_ownership in {args.db}")

if __name__ == "__main__":
    main()
