# features/statistical_features.py
"""
Computes statistical stretch/mean-reversion features from sep_base_common
with full coverage (no NULLs).

Features:
- close_zscore_21d: (close - mean_21) / std_21 with expanding fallbacks
- ret_1d_zscore_21d: ret_1d / std_21(ret_1d) with expanding fallback
- ret_1d_rank_21d: fast approximation via logistic CDF of z-score (0..1)
- price_pct_from_rolling_max_21d: (close - rolling_max_21) / rolling_max_21
- mean_reversion_flag: 1 if close_zscore_21d < -2.0 else 0

Run:
  python features/statistical_features.py --db-path data/kairos.duckdb
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
    CREATE OR REPLACE TABLE feat_stat AS
    WITH base AS (
      SELECT
        ticker, date, close,
        LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS close_lag1
      FROM sep_base_common
    ),
    rets AS (
      SELECT
        *,
        CASE
          WHEN close_lag1 IS NULL OR close_lag1 = 0 THEN 0
          ELSE (close / close_lag1) - 1
        END AS ret_1d
      FROM base
    ),
    roll AS (
      SELECT
        *,
        -- rolling 21 (rows 20 preceding) + expanding fallbacks
        AVG(close)   OVER (PARTITION BY ticker ORDER BY date
                           ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS mean21_close,
        STDDEV(close) OVER (PARTITION BY ticker ORDER BY date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std21_close,
        AVG(close)   OVER (PARTITION BY ticker ORDER BY date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS mean_exp_close,
        STDDEV(close) OVER (PARTITION BY ticker ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS std_exp_close,

        STDDEV(ret_1d) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std21_ret,
        STDDEV(ret_1d) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS std_exp_ret,

        MAX(close) OVER (PARTITION BY ticker ORDER BY date
                         ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS max21_close,
        MAX(close) OVER (PARTITION BY ticker ORDER BY date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS max_exp_close
      FROM rets
    ),
    eff AS (
      SELECT
        *,
        -- effective moments with safe epsilons (never NULL/zero)
        COALESCE(mean21_close, mean_exp_close) AS mean_close_eff,
        COALESCE(NULLIF(std21_close, 0), NULLIF(std_exp_close, 0), 1e-12) AS std_close_eff,
        COALESCE(NULLIF(std21_ret, 0),  NULLIF(std_exp_ret, 0),  1e-12)   AS std_ret_eff,
        COALESCE(max21_close, max_exp_close, close) AS max_close_eff
      FROM roll
    )
    SELECT
      ticker,
      date,

      -- z-score of price
      (close - mean_close_eff) / std_close_eff AS close_zscore_21d,

      -- z-score of 1-day return
      ret_1d / std_ret_eff AS ret_1d_zscore_21d,

      -- fast rolling percentile proxy via logistic CDF of z-score
      1.0 / (1.0 + EXP(-1.702 * (ret_1d / std_ret_eff))) AS ret_1d_rank_21d,

      -- distance from rolling max
      CASE
        WHEN max_close_eff = 0 THEN 0
        ELSE (close - max_close_eff) / max_close_eff
      END AS price_pct_from_rolling_max_21d,

      -- simple mean reversion flag
      CASE WHEN (close - mean_close_eff) / std_close_eff < -2.0 THEN 1 ELSE 0 END AS mean_reversion_flag

    FROM eff;
    """)

    con.close()
    print("✅ feat_stat built with full coverage.")

if __name__ == "__main__":
    main()
