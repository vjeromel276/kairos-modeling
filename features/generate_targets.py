"""
generate_targets.py

Creates forward-looking regression and classification targets from DuckDB sep_base.

Targets:
- ret_1d_f: Next day return
- ret_5d_f: 5-day forward return
- label_5d_up: Classification label: 1 if ret_5d_f > 0 else 0

Input:
    DuckDB table: sep_base

Output:
    DuckDB table: feat_targets

To run:
    python scripts/features/generate_targets.py --db data/kairos.duckdb
"""

import duckdb
import pandas as pd
import argparse

def generate_targets(con):
    df = con.execute("""
        SELECT ticker, date, close
        FROM sep_base
        ORDER BY ticker, date
    """).fetchdf()

    df["ret_1d_f"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df["ret_5d_f"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
    df["label_5d_up"] = (df["ret_5d_f"] > 0).astype("Int8")
    df = df.dropna()

    return df[["ticker", "date", "ret_1d_f", "ret_5d_f", "label_5d_up"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_targets")

    df_targets = generate_targets(con)
    con.execute("CREATE TABLE feat_targets AS SELECT * FROM df_targets")

    print(f"✅ Saved {len(df_targets):,} rows to feat_targets in {args.db}")

if __name__ == "__main__":
    main()
