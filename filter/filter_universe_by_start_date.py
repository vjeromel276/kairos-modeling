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

query = f"""
-- Step 1: Identify tickers in sep_base with data on or before the cutoff
WITH tickers_alive_by_cutoff AS (
    SELECT DISTINCT ticker
    FROM sep_base
    WHERE date <= DATE '{CUTOFF_DATE}'
),

-- Step 2: Intersect with the current midcap universe (filtered & vetted)
filtered_universe AS (
    SELECT DISTINCT ticker
    FROM mid_cap_2025_07_15
)

-- Step 3: Create the new modeling universe
CREATE OR REPLACE TABLE midcap_{args.year}_universe AS
SELECT f.ticker
FROM filtered_universe f
JOIN tickers_alive_by_cutoff t USING (ticker);
"""

con.execute(query)
print(f"✅ Saved: midcap_{args.year}_universe in DuckDB ({DB_PATH})")
