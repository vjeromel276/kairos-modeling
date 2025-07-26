#!/usr/bin/env python3
"""
Plot universe survival decay: how many of today's tickers were alive in each past year.
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "data/kairos.duckdb"
con = duckdb.connect(DB_PATH)

# Step 1: Get first date per ticker in the current universe
query = """
WITH current_universe AS (
    SELECT DISTINCT ticker FROM sep_base
),
first_dates AS (
    SELECT
        s.ticker,
        MIN(s.date) AS first_date
    FROM sep_base s
    JOIN current_universe u USING (ticker)
    GROUP BY s.ticker
),
years AS (
    SELECT range AS year
    FROM range(1997, 2026)  -- includes 2025
)
SELECT
    y.year,
    COUNT(fd.ticker) AS n_alive
FROM years y
LEFT JOIN first_dates fd
    ON fd.first_date <= MAKE_DATE(y.year, 1, 1)
GROUP BY y.year
ORDER BY y.year;

"""

df = con.execute(query).fetchdf()

# Plot it
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['n_alive'], marker='o')
plt.title('Ticker Survival Curve: Active Tickers by Year')
plt.xlabel('Year')
plt.ylabel('Number of Current Tickers Active as of That Year')
plt.grid(True)
plt.axhline(y=900, color='green', linestyle='--', label='900-ticker mark')
plt.axhline(y=600, color='orange', linestyle='--', label='600-ticker mark')
plt.tight_layout()
plt.legend()
plt.show()
