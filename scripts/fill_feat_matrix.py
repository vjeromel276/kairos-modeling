#!/usr/bin/env python3
"""
Phase 7.5: fill_feat_matrix.py (chunked version)

Builds a memory-safe version of `feat_matrix_filled` by processing tickers in chunks.
Fills NaNs via ffill + bfill per ticker, EXCLUDING target columns.

Usage:
    python scripts/features/fill_feat_matrix.py --db data/kairos.duckdb
"""

import argparse
import duckdb
import pandas as pd
from tqdm import tqdm

EXCLUDE_COLS = {
    'ticker', 'date',
    'ret_1d', 'ret_5d', 'ret_21d',
    'ret_1d_f', 'ret_5d_f', 'ret_21d_f'
}
TARGET_COL = 'ret_5d_f'
CHUNK_SIZE = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    print("Fetching ticker list...")
    tickers = con.execute("SELECT DISTINCT ticker FROM feat_matrix ORDER BY ticker").fetchdf()['ticker'].tolist()

    print("Dropping existing feat_matrix_filled if exists...")
    con.execute("DROP TABLE IF EXISTS feat_matrix_filled")
    con.execute("CREATE TABLE feat_matrix_filled AS SELECT * FROM feat_matrix WHERE false")  # empty shell

    for i in tqdm(range(0, len(tickers), CHUNK_SIZE), desc="Filling in chunks"):
        chunk = tickers[i:i + CHUNK_SIZE]
        query = f"SELECT * FROM feat_matrix WHERE ticker IN ({','.join(['?'] * len(chunk))}) ORDER BY ticker, date"
        df = con.execute(query, chunk).fetchdf()

        id_cols = ['ticker', 'date']
        fillable_cols = [c for c in df.columns if c not in EXCLUDE_COLS | {TARGET_COL} and pd.api.types.is_numeric_dtype(df[c])]

        df[fillable_cols] = (
            df.groupby("ticker")[fillable_cols]
              .transform(lambda g: g.ffill().bfill())
        )

        con.register("df_chunk", df)
        con.execute("INSERT INTO feat_matrix_filled SELECT * FROM df_chunk")

    print("✔ feat_matrix_filled rebuilt successfully in chunks.")

if __name__ == "__main__":
    main()