# technical_features.py
# Computes common technical indicators from sep_base_common with full coverage (no NULLs)

import argparse
import duckdb

def parse_args():
    p = argparse.ArgumentParser(description="Build technical indicator features")
    p.add_argument(
        "--db-path", default="data/kairos.duckdb",
        help="Path to DuckDB file containing sep_base_common"
    )
    return p.parse_args()

def main():
    args = parse_args()
    conn = duckdb.connect(database=args.db_path)

    conn.execute("""
    CREATE OR REPLACE TABLE feat_technical AS
    WITH base AS (
      SELECT
        ticker, date, open, high, low, close, volume,
        LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS lag_close
      FROM sep_base_common
    ),
    step2 AS (
      SELECT
        *,
        -- gains/losses from precomputed lag_close (no nested windows)
        COALESCE(GREATEST(close - lag_close, 0), 0) AS gain,
        COALESCE(GREATEST(lag_close - close, 0), 0) AS loss,

        -- rolling 20 + expanding fallbacks
        AVG(close)  OVER (PARTITION BY ticker ORDER BY date
                          ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
        STDDEV(close) OVER (PARTITION BY ticker ORDER BY date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS stddev_20,

        AVG(close)  OVER (PARTITION BY ticker ORDER BY date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS exp_mean_close,
        STDDEV(close) OVER (PARTITION BY ticker ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS exp_std_close,

        -- RSI(14) components (now safe: no nested window fns)
        AVG(gain) OVER (PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain_14,
        AVG(loss) OVER (PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss_14,

        -- stochastic range (14)
        MAX(high) OVER (PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS high_14,
        MIN(low)  OVER (PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS low_14
      FROM base
    ),
    final AS (
      SELECT
        *,
        -- effective (never-null) SMA/STD for Bollinger
        COALESCE(sma_20, exp_mean_close) AS sma_20_eff,
        COALESCE(NULLIF(stddev_20, 0), NULLIF(exp_std_close, 0), 1e-12) AS stddev_20_eff,

        -- Chaikin money-flow multiplier; safe when high==low
        CASE
          WHEN high = low THEN 0.0
          ELSE ((close - low) - (high - close)) / (high - low)
        END AS mf_mult,

        -- Stoch %K with neutral fallback when range==0
        CASE
          WHEN (high_14 = low_14) OR (high_14 IS NULL) OR (low_14 IS NULL) THEN 50
          ELSE 100 * (close - low_14) / NULLIF(high_14 - low_14, 0)
        END AS stoch_k_14_3
      FROM step2
    )
    SELECT
      ticker,
      date,

      -- RSI(14): neutral when both avg gains/losses are zero, 100 if no losses
      CASE
        WHEN avg_gain_14 = 0 AND avg_loss_14 = 0 THEN 50
        WHEN avg_loss_14 = 0 THEN 100
        ELSE 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
      END AS rsi_14,

      -- Bollinger Bands (20) + %B with safe denominator
      sma_20_eff + 2 * stddev_20_eff AS bb_upper_20,
      sma_20_eff - 2 * stddev_20_eff AS bb_lower_20,
      CASE
        WHEN stddev_20_eff = 0 THEN 0.5
        ELSE (close - (sma_20_eff - 2 * stddev_20_eff)) / (4 * stddev_20_eff)
      END AS bb_pct_b,

      -- OBV (cumulative); first row contributes 0
      SUM(
        CASE
          WHEN close > lag_close THEN volume
          WHEN close < lag_close THEN -volume
          ELSE 0
        END
      ) OVER (PARTITION BY ticker ORDER BY date
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS obv,

      -- Chaikin Money Flow (21d)
      SUM(mf_mult * volume) OVER (
        PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
      ) AS chaikin_mf_21,

      -- Stochastic %K and %D (14,3)
      stoch_k_14_3,
      AVG(stoch_k_14_3) OVER (
        PARTITION BY ticker ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
      ) AS stoch_d_14_3

    FROM final;
    """)

    conn.close()
    print("✅ feat_technical built.")

if __name__ == "__main__":
    main()
