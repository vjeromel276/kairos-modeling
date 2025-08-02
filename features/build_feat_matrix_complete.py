#!/usr/bin/env python3
"""
Memory-efficient feature matrix builder.
Supports specifying a custom universe table.
"""

import duckdb
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, help="Label year for output table (used in name only)")
parser.add_argument('--universe', type=str, required=True, help="Universe table name in DuckDB")
parser.add_argument('--full', action='store_true', help="Join all feat_* tables incrementally in DuckDB")
args = parser.parse_args()

DB_PATH = "data/kairos.duckdb"
UNIVERSE_TABLE = args.universe
OUTPUT_TABLE = f"feat_matrix_complete_{args.year or 'custom'}"

con = duckdb.connect(DB_PATH)

print(f"📦 Building feature matrix {OUTPUT_TABLE} using universe '{UNIVERSE_TABLE}'")

if args.full:
    print("🔍 Finding feature tables...")
    all_tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    exclude_prefixes = ("feat_matrix", "feat_targets")
    feat_tables = [
        t for t in all_tables
        if t.startswith("feat_") and not t.startswith(exclude_prefixes)
    ]

    if not feat_tables:
        raise RuntimeError("❌ No feat_* tables found in DuckDB.")

    print(f"🔍 Found feature tables: {feat_tables}")

    # Start with first table joined to universe
    first_table = feat_tables[0]
    print(f"📥 Starting with: {first_table}")

    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS
        SELECT f.*
        FROM {first_table} f
        JOIN {UNIVERSE_TABLE} u USING (ticker)
        WHERE f.ticker IS NOT NULL AND f.date IS NOT NULL
    """)
    print(f"✅ Initialized {OUTPUT_TABLE} with {first_table}")

    # Join others
    for t in feat_tables[1:]:
        print(f"➕ Joining: {t}")
        con.execute(f"""
            CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
            SELECT *
            FROM {OUTPUT_TABLE}
            LEFT JOIN (
                SELECT * FROM {t}
            ) AS {t}
            USING (ticker, date)
        """)
        print(f"✅ Joined: {t}")

else:
    print("🔍 Building minimal base feature matrix from sep_base...")
    df = con.execute(f"""
        SELECT s.*
        FROM sep_base s
        JOIN {UNIVERSE_TABLE} u USING (ticker)
    """).fetchdf()

    df = df.sort_values(["ticker", "date"])
    df["log_return"] = np.log(df["closeadj"] / df.groupby("ticker")["closeadj"].shift(1))
    df["volume_z"] = df.groupby("ticker")["volume"].transform(lambda x: (x - x.mean()) / x.std())
    df["price_norm"] = df["closeadj"] / df.groupby("ticker")["closeadj"].transform("max")
    df = df[["ticker", "date", "log_return", "volume_z", "price_norm"]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.register("df", df)
    con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df")

    print(f"✅ Saved minimal matrix to {OUTPUT_TABLE} ({len(df):,} rows)")

print(f"🎯 Done: Feature matrix {OUTPUT_TABLE} is ready.")
