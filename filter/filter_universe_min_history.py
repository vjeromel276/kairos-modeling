#!/usr/bin/env python3
"""
Filter tickers with sufficient trading history that meet mid/large/mega cap criteria.
Creates a new universe table: midcap_long_history_universe
"""

import duckdb
import pandas as pd

DB_PATH = "data/kairos.duckdb"
MIN_DAYS = 252  # Minimum trading days required

con = duckdb.connect(DB_PATH)

print(f"Building midcap_long_history_universe with ≥ {MIN_DAYS} trading days...")

query = f"""
WITH base_universe AS (
    SELECT ticker
    FROM ticker_metadata_view
    WHERE 
        scalemarketcap IN ('4 - Mid', '5 - Large', '6 - Mega') AND
        LOWER(category) LIKE '%common stock%' AND
        volumeavg1m >= 2_000_000
),
sufficient_history AS (
    SELECT ticker
    FROM sep_base
    WHERE date >= DATE '1999-01-01'
    GROUP BY ticker
    HAVING COUNT(DISTINCT date) >= {MIN_DAYS}
)
SELECT DISTINCT b.ticker
FROM base_universe b
JOIN sufficient_history s USING (ticker)
"""

# Execute and fetch
df = con.execute(query).fetchdf()

# Save to DuckDB
table_name = "midcap_long_history_universe"
con.execute(f"DROP TABLE IF EXISTS {table_name}")
con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

print(f"✅ Saved: {table_name} in DuckDB ({DB_PATH})")
con.close()
print(f"✅ {len(df)} tickers in long-history universe")
print("Done.")
