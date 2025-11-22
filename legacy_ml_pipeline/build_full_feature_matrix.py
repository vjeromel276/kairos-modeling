"""
build_full_feature_matrix.py

Joins all feat_* tables + targets into a training-ready matrix.
Filters to a user-specified universe CSV.

Usage:
    python scripts/features/build_full_feature_matrix.py \
        --db data/kairos.duckdb \
        --date 2025-07-13 \
        --universe scripts/sep_dataset/feature_sets/midcap_and_up_ticker_universe_2025-07-13.csv
"""

import duckdb
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--date", required=True, help="Date string (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Path to ticker universe CSV")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    # Load ticker universe dynamically
    print(f"📂 Loading ticker universe: {args.universe}")
    universe_df = pd.read_csv(args.universe)
    con.execute("CREATE OR REPLACE TEMP TABLE ticker_universe AS SELECT * FROM universe_df")

    # Discover feat_* tables
    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    feature_tables = [t for t in tables if t.startswith("feat_")]

    if not feature_tables:
        raise RuntimeError("❌ No feat_* tables found in DuckDB")

    print(f"🔍 Found {len(feature_tables)} feature tables: {feature_tables}")

    # Start with the first table
    base = feature_tables[0]
    join_query = f"SELECT * FROM {base}"

    for t in feature_tables[1:]:
        cols = con.execute(f"PRAGMA table_info('{t}')").fetchdf()
        dupes = [c for c in cols["name"] if c in ("ticker", "date")]
        select_cols = ", ".join([f"{t}.{c}" for c in cols["name"] if c not in dupes])
        join_query = f"""
            SELECT *
            FROM ({join_query}) AS base
            LEFT JOIN (
                SELECT ticker, date, {select_cols}
                FROM {t}
            ) AS {t}
            USING (ticker, date)
        """

    # Final query with universe filter
    final_query = f"""
        SELECT *
        FROM ({join_query}) AS merged
        INNER JOIN ticker_universe USING (ticker)
        WHERE ticker IS NOT NULL AND date IS NOT NULL
    """

    df = con.execute(final_query).fetchdf()
    print(f"✅ Final matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Save to table + snapshot
    con.execute("DROP TABLE IF EXISTS feat_matrix")
    con.execute("CREATE TABLE feat_matrix AS SELECT * FROM df")

    output_path = Path(f"scripts/feature_matrices/{args.date}_full_feature_matrix.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"💾 Saved feature matrix to: {output_path}")

if __name__ == "__main__":
    main()
