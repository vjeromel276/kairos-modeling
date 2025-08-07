# technical_features.py
# Computes common technical indicators from sep_base_common

import argparse
import duckdb


def parse_args():
    parser = argparse.ArgumentParser(description="Build technical indicators features")
    parser.add_argument(
        "--db-path", default="data/kairos.duckdb",
        help="Path to DuckDB file containing sep_base_common"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = duckdb.connect(database=args.db_path)

    # Create or replace the technical features table
    conn.execute("""
    CREATE OR REPLACE TABLE feat_technical AS
    SELECT
        ticker,
        date,
        -- RSI 14
        CASE WHEN avg_loss_14 = 0 THEN 100
             ELSE 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
        END AS rsi_14,
        -- Bollinger Bands (20)
        sma_20 + 2 * stddev_20      AS bb_upper_20,
        sma_20 - 2 * stddev_20      AS bb_lower_20,
        (close - (sma_20 - 2*stddev_20)) / NULLIF((4*stddev_20),0) AS bb_pct_b,
        -- On-Balance Volume (OBV)
        SUM(CASE WHEN close > lag_close THEN volume
                 WHEN close < lag_close THEN -volume
                 ELSE 0 END
        ) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS obv,
        -- Chaikin Money Flow (21-day)
        SUM(((close - low) - (high - close)) / NULLIF((high - low),0) * volume)
        OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS chaikin_mf_21,
        -- Stochastic Oscillator (14,3)
        100 * (close - low_14) / NULLIF((high_14 - low_14),0) AS stoch_k_14_3,
        AVG(100 * (close - low_14) / NULLIF((high_14 - low_14),0))
        OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS stoch_d_14_3
    FROM (
        SELECT
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume,
            -- lagged close for returns and sign
            LAG(close) OVER w            AS lag_close,
            -- gains/losses
            GREATEST(close - LAG(close) OVER w, 0)    AS gain,
            GREATEST(LAG(close) OVER w - close, 0)    AS loss,
            -- RSI running averages
            AVG(GREATEST(close - LAG(close) OVER w, 0))
                OVER (w ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain_14,
            AVG(GREATEST(LAG(close) OVER w - close, 0))
                OVER (w ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss_14,
            -- Bollinger rolling stats
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)    AS sma_20,
            STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS stddev_20,
            -- Stochastic highest high / lowest low
            MAX(high) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS high_14,
            MIN(low)  OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS low_14
        FROM sep_base_common
        WINDOW w AS (PARTITION BY ticker ORDER BY date)
    ) sub;
    """)

    conn.close()
    print("✅ feat_technical built.")

if __name__ == "__main__":
    main()
