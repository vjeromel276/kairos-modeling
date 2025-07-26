#!/usr/bin/env python3
"""
Generate forward return targets for a given feature matrix year.
Computes ret_1d_f, ret_5d_f, ret_21d_f and writes to feat_matrix_targets_<year>.
"""

import duckdb
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True, help='Universe year (e.g., 2008)')
args = parser.parse_args()

DB_PATH = "data/kairos.duckdb"
INPUT_TABLE = f"feat_matrix_complete_{args.year}"
OUTPUT_TABLE = f"feat_matrix_targets_{args.year}"

print(f"📦 Loading {INPUT_TABLE} + closeadj from sep_base...")
con = duckdb.connect(DB_PATH)

# Join in closeadj from sep_base
df = con.execute(f"""
    SELECT f.*, s.closeadj
    FROM {INPUT_TABLE} f
    JOIN sep_base s USING (ticker, date)
""").fetchdf()

print("🧮 Computing forward returns...")

df = df.sort_values(["ticker", "date"])
df["ret_1d_f"] = df.groupby("ticker")["closeadj"].transform(lambda x: np.log(x.shift(-1) / x))
df["ret_5d_f"] = df.groupby("ticker")["closeadj"].transform(lambda x: np.log(x.shift(-5) / x))
df["ret_21d_f"] = df.groupby("ticker")["closeadj"].transform(lambda x: np.log(x.shift(-21) / x))

df = df.drop(columns=["closeadj"])  # optional: remove to keep feature matrix clean
df = df.dropna(subset=["ret_1d_f", "ret_5d_f", "ret_21d_f"])

# Save result
print(f"💾 Saving to {OUTPUT_TABLE}...")
con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
con.register("df", df)
con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df")

print(f"✅ Done. Targets saved to {OUTPUT_TABLE}")
