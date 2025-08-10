# features/price_shape_features.py
"""
Extract candlestick shape & gap features from sep_base_common with full coverage.

Features:
- body_size, upper_wick, lower_wick, candle_range
- body_pct_of_range, upper_wick_pct, lower_wick_pct
- gap_open, gap_pct

Run:
  python features/price_shape_features.py --db-path data/kairos.duckdb
"""
import argparse
import duckdb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="data/kairos.duckdb",
                   help="Path to DuckDB database")
    return p.parse_args()

def main():
    args = parse_args()
    con = duckdb.connect(args.db_path)

    con.execute("""
    CREATE OR REPLACE TABLE feat_price_shape AS
    WITH base AS (
      SELECT
        ticker, date, open, high, low, close,
        LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
      FROM sep_base_common
    ),
    calc AS (
      SELECT
        *,
        ABS(close - open) AS body_size,
        GREATEST(high - GREATEST(open, close), 0) AS upper_wick,
        GREATEST(LEAST(open, close) - low, 0)     AS lower_wick,
        (high - low) AS candle_range
      FROM base
    )
    SELECT
      ticker,
      date,
      body_size,
      upper_wick,
      lower_wick,
      candle_range,
      CASE WHEN candle_range = 0 THEN 0 ELSE body_size  / candle_range END AS body_pct_of_range,
      CASE WHEN candle_range = 0 THEN 0 ELSE upper_wick / candle_range END AS upper_wick_pct,
      CASE WHEN candle_range = 0 THEN 0 ELSE lower_wick / candle_range END AS lower_wick_pct,
      COALESCE(open - prev_close, 0) AS gap_open,
      CASE
        WHEN prev_close IS NULL OR prev_close = 0 THEN 0
        ELSE (open - prev_close) / prev_close
      END AS gap_pct
    FROM calc;
    """)

    con.close()
    print("✅ feat_price_shape built with full coverage.")

if __name__ == "__main__":
    main()
