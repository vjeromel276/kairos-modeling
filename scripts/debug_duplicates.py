#!/usr/bin/env python3
"""
debug_duplicates.py

This diagnostic script checks ALL major feature tables for duplicate
(ticker, date) rows — the #1 cause of matrix blowups and exploding backtests.

Run:
    python scripts/debug_duplicates.py --db data/kairos.duckdb
"""

import duckdb
import argparse

TABLES_TO_CHECK = [
    "sep_base_academic",
    "feat_price_action",
    "feat_price_shape",
    "feat_stat",
    "feat_trend",
    "feat_volume_volatility",
    "feat_targets",
    "feat_composite_academic",
    "feat_institutional_academic",
    "feat_composite_long",
    "feat_composite_v3",
    "feat_composite_v31",
    "feat_composite_v32",
    "feat_vol_sizing",
]


def check_table(con, table):
    print(f"\n🔍 Checking table: {table}")

    # Quick existence check
    exists = con.execute(f"SELECT 1 FROM duckdb_tables WHERE table_name = '{table}'").fetchone()
    if not exists:
        print(f"   ❌ Table does not exist: {table}")
        return

    # Count duplicates
    dup_query = f"""
        SELECT COUNT(*) AS dup_count
        FROM (
            SELECT ticker, date, COUNT(*) as c
            FROM {table}
            GROUP BY ticker, date
            HAVING c > 1
        )
    """

    dup_count = con.execute(dup_query).fetchone()[0]

    if dup_count == 0:
        print(f"   ✔ No duplicates in (ticker, date) — good!")
        return

    print(f"   ⚠ Found {dup_count:,} duplicate (ticker,date) rows!")

    # Show sample duplicates
    sample_query = f"""
        SELECT ticker, date, COUNT(*) as c
        FROM {table}
        GROUP BY ticker, date
        HAVING c > 1
        LIMIT 20
    """
    sample = con.execute(sample_query).fetchdf()
    print("\n   🔎 Sample duplicates:")
    print(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    print("\n=========================================")
    print("🔍 DUPLICATE CHECK ACROSS FEATURE TABLES")
    print("=========================================\n")

    for table in TABLES_TO_CHECK:
        check_table(con, table)

    print("\n=========================================")
    print("🔍 Duplicate check complete.")
    print("=========================================")


if __name__ == "__main__":
    main()
