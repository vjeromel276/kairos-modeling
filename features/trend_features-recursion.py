# features/trend_features.py
"""
Trend features from sep_base_common with full coverage.

Features:
- SMA (5, 21)
- EMA (12, 26)
- price_vs_sma_21
- sma_21_slope (5-day diff)
- MACD (ema12-ema26), MACD signal (EMA 9 of MACD), MACD histogram

Run:
  python features/trend_features.py --db-path data/kairos.duckdb
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
    CREATE OR REPLACE TABLE feat_trend AS
    WITH RECURSIVE
    -- Base rows with row numbers per ticker
    base AS (
      SELECT
        ticker, date, close,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS rn
      FROM sep_base_common
    ),
    -- EMA coefficients
    params AS (
      SELECT
        2.0/13.0 AS a12,   -- EMA(12)
        2.0/27.0 AS a26,   -- EMA(26)
        2.0/10.0 AS a9     -- EMA(9) for MACD signal
    ),
    -- Recursive EMA(12) and EMA(26) over CLOSE per ticker
    ema_calc AS (
      -- anchors: rn=1 -> EMA = first close
      SELECT b.ticker, b.date, b.rn, b.close,
             b.close AS ema12,
             b.close AS ema26
      FROM base b
      WHERE b.rn = 1
      UNION ALL
      SELECT b.ticker, b.date, b.rn, b.close,
             (p.a12 * b.close + (1 - p.a12) * e.ema12) AS ema12,
             (p.a26 * b.close + (1 - p.a26) * e.ema26) AS ema26
      FROM base b
      JOIN ema_calc e
        ON b.ticker = e.ticker AND b.rn = e.rn + 1
      CROSS JOIN params p
    ),
    -- MACD from EMAs
    macd_base AS (
      SELECT
        ticker, date, rn, close,
        ema12, ema26,
        (ema12 - ema26) AS macd
      FROM ema_calc
    ),
    -- Recursive EMA(9) of MACD (signal line)
    macd_signal_calc AS (
      -- anchor: rn=1 -> signal = first macd
      SELECT m.ticker, m.date, m.rn, m.close, m.ema12, m.ema26, m.macd,
             m.macd AS macd_signal
      FROM macd_base m
      WHERE m.rn = 1
      UNION ALL
      SELECT m.ticker, m.date, m.rn, m.close, m.ema12, m.ema26, m.macd,
             (p.a9 * m.macd + (1 - p.a9) * s.macd_signal) AS macd_signal
      FROM macd_base m
      JOIN macd_signal_calc s
        ON m.ticker = s.ticker AND m.rn = s.rn + 1
      CROSS JOIN params p
    ),
    -- SMAs with window averages (DuckDB averages over available rows, so no NULLs)
    smas AS (
      SELECT
        ticker, date, rn, close,
        AVG(close) OVER (PARTITION BY ticker ORDER BY date
                         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS sma_5,
        AVG(close) OVER (PARTITION BY ticker ORDER BY date
                         ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS sma_21
      FROM base
    )
    SELECT
      s.ticker,
      s.date,
      s.sma_5,
      s.sma_21,
      ms.ema12 AS ema_12,
      ms.ema26 AS ema_26,
      CASE WHEN s.sma_21 IS NULL OR s.sma_21 = 0 THEN 0
           ELSE (s.close - s.sma_21) / s.sma_21 END AS price_vs_sma_21,
      COALESCE(
        s.sma_21 - LAG(s.sma_21, 5) OVER (PARTITION BY s.ticker ORDER BY s.date),
        0
      ) AS sma_21_slope,
      ms.macd,
      ms.macd_signal,
      (ms.macd - ms.macd_signal) AS macd_hist
    FROM smas s
    JOIN macd_signal_calc ms
      ON s.ticker = ms.ticker AND s.rn = ms.rn
    ORDER BY s.ticker, s.date;
    """)

    con.close()
    print("✅ feat_trend built with full coverage.")

if __name__ == "__main__":
    main()
