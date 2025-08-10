# features/volume_volatility_features.py
"""
Volume & volatility features from sep_base_common with full coverage.

Features:
- dollar_volume = close * volume
- vol_zscore_21d  (with expanding fallback + epsilon)
- volume_pct_change_1d  (safe when prev volume is 0/missing)
- ret_std_5d, ret_std_21d  (with expanding fallback)
- atr_14d  (True Range with prev close; 14-day SMA with expanding fallback)

Run:
  python features/volume_volatility_features.py --db-path data/kairos.duckdb
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
    CREATE OR REPLACE TABLE feat_volume_volatility AS
    WITH base AS (
      SELECT
        ticker, date, close, high, low, volume,
        LAG(close)  OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
        LAG(volume) OVER (PARTITION BY ticker ORDER BY date) AS prev_volume
      FROM sep_base_common
    ),
    rets AS (
      SELECT
        *,
        -- simple return; neutral when prev_close missing/0
        CASE WHEN prev_close IS NULL OR prev_close = 0 THEN 0
             ELSE (close / prev_close) - 1 END AS ret_1d,
        -- volume % change; neutral when prev_volume missing/0
        CASE WHEN prev_volume IS NULL OR prev_volume = 0 THEN 0
             ELSE (volume / prev_volume) - 1 END AS volume_pct_change_1d,
        -- True Range using Wilder definition with prev close
        GREATEST(
          high - low,
          ABS(high - COALESCE(prev_close, close)),
          ABS(low  - COALESCE(prev_close, close))
        ) AS true_range
      FROM base
    ),
    roll AS (
      SELECT
        *,
        -- volume rolling moments (21) + expanding fallbacks
        AVG(volume)  OVER (PARTITION BY ticker ORDER BY date
                           ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS mean21_vol,
        STDDEV(volume) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std21_vol,
        AVG(volume)  OVER (PARTITION BY ticker ORDER BY date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS mean_exp_vol,
        STDDEV(volume) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS std_exp_vol,

        -- return volatility (5, 21) + expanding fallback
        STDDEV(ret_1d) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS std5_ret,
        STDDEV(ret_1d) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std21_ret,
        STDDEV(ret_1d) OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS std_exp_ret,

        -- ATR(14) simple moving average + expanding fallback
        AVG(true_range) OVER (PARTITION BY ticker ORDER BY date
                              ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr14_sma,
        AVG(true_range) OVER (PARTITION BY ticker ORDER BY date
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS atr_exp
      FROM rets
    ),
    eff AS (
      SELECT
        *,
        -- effective (never-null) denominators with epsilon
        COALESCE(mean21_vol, mean_exp_vol) AS mean_vol_eff,
        COALESCE(NULLIF(std21_vol,0), NULLIF(std_exp_vol,0), 1e-12) AS std_vol_eff,
        COALESCE(NULLIF(std5_ret,0),  NULLIF(std_exp_ret,0), 1e-12) AS std5_ret_eff,
        COALESCE(NULLIF(std21_ret,0), NULLIF(std_exp_ret,0), 1e-12) AS std21_ret_eff,
        COALESCE(atr14_sma, atr_exp, 0) AS atr14_eff
      FROM roll
    )
    SELECT
      ticker,
      date,
      (close * volume) AS dollar_volume,
      -- z-score of volume w/ safe denom
      (volume - mean_vol_eff) / std_vol_eff AS vol_zscore_21d,
      volume_pct_change_1d,
      std5_ret_eff  AS ret_std_5d,
      std21_ret_eff AS ret_std_21d,
      atr14_eff     AS atr_14d
    FROM eff;
    """)

    con.close()
    print("✅ feat_volume_volatility built with full coverage.")

if __name__ == "__main__":
    main()
