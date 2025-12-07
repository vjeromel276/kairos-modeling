#!/usr/bin/env python3
"""
Create training-safe versions of selected feat_* tables.
Each output table is suffixed with _filled (e.g., feat_trend_filled).
Does not modify original tables. Processes by ticker in chunks.

Usage:
    python scripts/features/ffill_bfill_feat_tables.py --db data/kairos.duckdb
"""
import argparse
import duckdb
import pandas as pd
from tqdm import tqdm

TARGET_SUFFIX = '_f'
CHUNK_SIZE = 100
FILL_TABLES = [
    "feat_price_action",
    "feat_price_shape",
    "feat_stat",
    "feat_trend",
    "feat_volume_volatility",
    "feat_institutional_academic",
    "feat_value",
    "feat_adv",
    "feat_vol_sizing",
    "feat_beta"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    for table in FILL_TABLES:
        filled_table = f"{table}_filled"
        print(f"\n▶ Processing {table} → {filled_table}")

        tickers = con.execute(f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker").fetchdf()['ticker'].tolist()
        con.execute(f"DROP TABLE IF EXISTS {filled_table}")
        con.execute(f"CREATE TABLE {filled_table} AS SELECT * FROM {table} WHERE false")

        sample = con.execute(f"SELECT * FROM {table} LIMIT 1").fetchdf()
        numeric_cols = [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c]) and not c.endswith(TARGET_SUFFIX)]

        for i in tqdm(range(0, len(tickers), CHUNK_SIZE), desc=f"Filling {table}"):
            chunk = tickers[i:i + CHUNK_SIZE]
            query = f"SELECT * FROM {table} WHERE ticker IN ({','.join(['?'] * len(chunk))}) ORDER BY ticker, date"
            df = con.execute(query, chunk).fetchdf()

            df[numeric_cols] = (
                df.groupby("ticker")[numeric_cols]
                  .transform(lambda g: g.ffill().bfill())
            )

            con.register("df_chunk", df)
            con.execute(f"INSERT INTO {filled_table} SELECT * FROM df_chunk")

        print(f"✔ Completed: {filled_table}")

if __name__ == "__main__":
    main()
