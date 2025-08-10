# features/price_action_features.py
"""
Generates price action and momentum-based features from sep_base_common
with full coverage (no NULLs, no dropped rows).

Features:
- ret_1d, ret_5d, ret_21d  (backward-looking simple returns)
- hl_ratio, co_ratio
- true_range (high - low)
- range_pct  ((high - low) / open)

Run:
  python features/price_action_features.py --db-path data/kairos.duckdb
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
    CREATE OR REPLACE TABLE feat_price_action AS
    WITH base AS (
      SELECT
        ticker, date, open, high, low, close,
        LAG(close, 1)  OVER w AS close_lag1,
        LAG(close, 5)  OVER w AS close_lag5,
        LAG(close, 21) OVER w AS close_lag21
      FROM sep_base_common
      WINDOW w AS (PARTITION BY ticker ORDER BY date)
    ),
    rets AS (
      SELECT
        *,
        -- Simple returns with neutral fallback (0) when lookback is missing/0
        CASE WHEN close_lag1  IS NULL OR close_lag1  = 0 THEN 0
             ELSE (close / close_lag1)  - 1 END AS ret_1d,
        CASE WHEN close_lag5  IS NULL OR close_lag5  = 0 THEN 0
             ELSE (close / close_lag5)  - 1 END AS ret_5d,
        CASE WHEN close_lag21 IS NULL OR close_lag21 = 0 THEN 0
             ELSE (close / close_lag21) - 1 END AS ret_21d
      FROM base
    )
    SELECT
      ticker,
      date,
      ret_1d,
      ret_5d,
      ret_21d,
      -- Ratios with safe divides
      CASE WHEN low  = 0 THEN 1 ELSE high / low  END AS hl_ratio,
      CASE WHEN open = 0 THEN 1 ELSE close / open END AS co_ratio,
      -- Ranges
      (high - low)                                           AS true_range,
      CASE WHEN open = 0 THEN 0 ELSE (high - low) / open END AS range_pct
    FROM rets;
    """)

    con.close()
    print("✅ feat_price_action built with full coverage.")

if __name__ == "__main__":
    main()
