#!/usr/bin/env python3
"""
Joins cross-sectional z-score and rank features back into a feature matrix.

Usage:
    python features/merge_cross_sectional.py \
      --db data/kairos.duckdb \
      --matrix feat_matrix_complete_1999 \
      --cs-table feat_cross_sectional
"""

import duckdb
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--matrix", required=True, help="Name of the feature matrix table")
    parser.add_argument("--cs-table", default="feat_cross_sectional", help="Cross-sectional features table")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    print(f"🔁 Merging cross-sectional features from {args.cs_table} into {args.matrix}...")

    con.execute(f"""
        CREATE OR REPLACE TABLE {args.matrix} AS
        SELECT m.*, x.*
        FROM {args.matrix} m
        LEFT JOIN {args.cs_table} x
        USING (ticker, date)
    """)

    print(f"✅ Matrix {args.matrix} now includes cross-sectional features.")

if __name__ == "__main__":
    main()
