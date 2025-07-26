#!/usr/bin/env python3
"""
Filter tickers from mid_cap_2025_07_15 that were active by a given start date.
Creates a new universe table: midcap_<year>_universe
"""

import duckdb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2008, help='Cutoff year (inclusive)')
args = parser.parse_args()

DB_PATH = "data/kairos.duckdb"
CUTOFF_DATE = f"{args.year}-01-01"

con = duckdb.connect(DB_PATH)

print(f"Building midcap_{args.year}_universe using cutoff {CUTOFF_DATE}...")

# Step 1: Get the list of tickers alive by cutoff and in the current midcap universe
alive_query = f"""
WITH tickers_alive_by_cutoff AS (
    SELECT DISTINCT ticker
    FROM sep_base
    WHERE date <= DATE '{CUTOFF_DATE}'
),
filtered_universe AS (
    SELECT DISTINCT ticker
    FROM mid_cap_2025_07_15
)
SELECT f.ticker
FROM filtered_universe f
JOIN tickers_alive_by_cutoff t USING (ticker);
"""

# Step 2: Run the query and save as new table
df = con.execute(alive_query).fetchdf()

# Step 3: Save to DuckDB table
table_name = f"midcap_{args.year}_universe"
con.execute(f"DROP TABLE IF EXISTS {table_name}")
con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

print(f"✅ Saved: {table_name} in DuckDB ({DB_PATH})")
con.close()
print(f"✅ {len(df)} tickers in {table_name} universe")
print(f"✅ Universe created for year {args.year} with cutoff date {CUTOFF_DATE}")
print("Done.")
