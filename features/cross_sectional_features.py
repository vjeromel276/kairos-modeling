#!/usr/bin/env python3
"""
Generate cross-sectional z-scores and ranks across the universe per day.

Takes the full feature matrix as input (feat_matrix_complete_<year>)
and outputs feat_cross_sectional table with _z and _rank features.
"""

import duckdb
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--db", required=True, help="Path to DuckDB database")
parser.add_argument("--matrix", required=True, help="Feature matrix table (e.g. feat_matrix_complete_1999)")
args = parser.parse_args()

def compute_cross_sectional_features(con, table):
    df = con.execute(f"SELECT * FROM {table}").fetchdf()

    df = df.sort_values(["date", "ticker"])
    df = df.dropna(subset=["date", "ticker"])
    
    feature_cols = [
        col for col in df.columns
        if col not in ("ticker", "date") and df[col].dtype in [np.float32, np.float64, np.float16, float]
    ]

    zscored = []
    for col in feature_cols:
        grouped = df.groupby("date")[col]
        df[f"{col}_cs_z"] = grouped.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
        df[f"{col}_cs_rank"] = grouped.rank(pct=True)

    df = df.dropna()

    # Save only z/rank features + keys
    keep = ["ticker", "date"] + [c for c in df.columns if c.endswith("_cs_z") or c.endswith("_cs_rank")]
    return df[keep]

def main():
    con = duckdb.connect(args.db)
    output_table = "feat_cross_sectional"

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    df = compute_cross_sectional_features(con, args.matrix)
    con.execute(f"CREATE TABLE {output_table} AS SELECT * FROM df")

    print(f"✅ Cross-sectional features written to {output_table}")

if __name__ == "__main__":
    main()
