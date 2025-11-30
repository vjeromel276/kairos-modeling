#!/usr/bin/env python3
"""
generate_targets.py  (patched for sep_base_academic)

Creates forward-looking regression and classification targets from the 
optimized academic dataset sep_base_academic.

Targets:
    - ret_1d_f: Next-day return
    - ret_5d_f: 5-day forward return
    - label_5d_up: Classification (1 if ret_5d_f > 0 else 0)

Usage:
    python scripts/generate_targets.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import argparse


def generate_targets(con):
    # Pull sorted data from optimized academic base
    df = con.execute("""
        SELECT ticker, date, close
        FROM sep_base_academic
        ORDER BY ticker, date
    """).fetchdf()

    # Forward returns
    df["ret_1d_f"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df["ret_5d_f"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1

    # Binary label (for ML classification later)
    df["label_5d_up"] = (df["ret_5d_f"] > 0).astype("Int8")

    df = df.dropna()

    return df[["ticker", "date", "ret_1d_f", "ret_5d_f", "label_5d_up"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    # Replace old target table
    con.execute("DROP TABLE IF EXISTS feat_targets")

    df_targets = generate_targets(con)

    # Save into DuckDB as its own table
    con.execute("CREATE TABLE feat_targets AS SELECT * FROM df_targets")

    print(f"✔ Saved {len(df_targets):,} rows to feat_targets in {args.db}")


if __name__ == "__main__":
    main()
