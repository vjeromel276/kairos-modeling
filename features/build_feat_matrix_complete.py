#!/usr/bin/env python3
"""
Builds full feature matrix for a given modeling universe year.
Inputs: sep_base, midcap_<year>_universe
Output: feat_matrix_complete_<year> (in DuckDB)
"""

import duckdb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True, help="Universe cutoff year (e.g., 2008 or 2015)")
args = parser.parse_args()

DB_PATH = "data/kairos.duckdb"
UNIVERSE_TABLE = f"midcap_{args.year}_universe"
OUTPUT_TABLE = f"feat_matrix_complete_{args.year}"

print(f"📦 Building feature matrix for universe starting in {args.year}...")

con = duckdb.connect(DB_PATH)

# Step 1: Filter SEP to selected tickers
print(f"🔍 Filtering sep_base to tickers in {UNIVERSE_TABLE}...")
filtered = con.execute(f"""
    SELECT s.*
    FROM sep_base s
    JOIN {UNIVERSE_TABLE} u USING (ticker)
""").fetchdf()

# Step 2: Feature engineering
# Replace this block with your actual feature generation logic
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.sort_values(['ticker', 'date'])
    df['log_return'] = np.log(df['closeadj'] / df.groupby('ticker')['closeadj'].shift(1))
    df['volume_z'] = df.groupby('ticker')['volume'].transform(lambda x: (x - x.mean()) / x.std())
    df['price_norm'] = df['closeadj'] / df.groupby('ticker')['closeadj'].transform('max')
    return df

df_feat = engineer_features(filtered)

# Step 3: Save to DuckDB
print(f"💾 Writing {OUTPUT_TABLE} to DuckDB...")
con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
con.register("df_feat", df_feat)
con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df_feat")

print(f"✅ Feature matrix {OUTPUT_TABLE} saved.")
