# features/trend_features.py
"""
Trend features from sep_base_common using finite-window (vectorized) EMAs.
Guarantees one row per (ticker, date) by joining on (ticker, rn).

Features:
- SMA (5, 21)
- EMA (12, 26) via finite-window exponential weights
- price_vs_sma_21
- sma_21_slope (diff vs 5 days ago)
- MACD, MACD signal (EMA9 of MACD), MACD histogram

Run:
  python features/trend_features.py --db-path data/kairos.duckdb
        [--ema12-span 12 --ema26-span 26 --signal-span 9 --k-mult 6]
"""
import argparse
import duckdb

def ew_weights(span: int, k_mult: int):
    a = 2.0 / (span + 1.0)
    K = int(k_mult * span)
    ws = [a * ((1.0 - a) ** k) for k in range(K + 1)]
    return ws, K

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="data/kairos.duckdb")
    p.add_argument("--ema12-span", type=int, default=12)
    p.add_argument("--ema26-span", type=int, default=26)
    p.add_argument("--signal-span", type=int, default=9)
    p.add_argument("--k-mult", type=int, default=6)   # 6≈99.5% mass, 8≈99.9%
    return p.parse_args()

def main():
    args = parse_args()
    con = duckdb.connect(args.db_path)

    # weights
    w12, K12 = ew_weights(args.ema12_span, args.k_mult)
    w26, K26 = ew_weights(args.ema26_span, args.k_mult)
    w9,  K9  = ew_weights(args.signal_span, args.k_mult)

    Kc = max(K12, K26)

    # lags for CLOSE up to Kc
    close_lag_cols = ",\n        ".join(
        [f"LAG(close, {k}) OVER w_rn AS close_lag{k}" for k in range(1, Kc + 1)]
    )

    # EMA12 terms
    ema12_num_terms = [f"close * {w12[0]}"] + [
        f"COALESCE(close_lag{k}, 0) * {w12[k]}" for k in range(1, K12 + 1)
    ]
    ema12_den_terms = [f"{w12[0]}"] + [
        f"CASE WHEN close_lag{k} IS NULL THEN 0 ELSE {w12[k]} END" for k in range(1, K12 + 1)
    ]
    ema12_num = " + ".join(ema12_num_terms)
    ema12_den = " + ".join(ema12_den_terms)

    # EMA26 terms
    ema26_num_terms = [f"close * {w26[0]}"] + [
        f"COALESCE(close_lag{k}, 0) * {w26[k]}" for k in range(1, K26 + 1)
    ]
    ema26_den_terms = [f"{w26[0]}"] + [
        f"CASE WHEN close_lag{k} IS NULL THEN 0 ELSE {w26[k]} END" for k in range(1, K26 + 1)
    ]
    ema26_num = " + ".join(ema26_num_terms)
    ema26_den = " + ".join(ema26_den_terms)

    # MACD lags/signal terms
    macd_lag_cols = ",\n        ".join(
        [f"LAG(macd, {k}) OVER w_rn AS macd_lag{k}" for k in range(1, K9 + 1)]
    )
    sig_num_terms = [f"macd * {w9[0]}"] + [
        f"COALESCE(macd_lag{k}, 0) * {w9[k]}" for k in range(1, K9 + 1)
    ]
    sig_den_terms = [f"{w9[0]}"] + [
        f"CASE WHEN macd_lag{k} IS NULL THEN 0 ELSE {w9[k]} END" for k in range(1, K9 + 1)
    ]
    sig_num = " + ".join(sig_num_terms)
    sig_den = " + ".join(sig_den_terms)

    sql = f"""
    CREATE OR REPLACE TABLE feat_trend AS
    WITH base AS (
      SELECT
        ticker, date, close,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS rn
      FROM sep_base_common
    ),
    -- SMAs (no nesting). Compute SMA_5 & SMA_21 first...
    smas0 AS (
      SELECT
        ticker, date, rn, close,
        AVG(close) OVER (PARTITION BY ticker ORDER BY rn
                         ROWS BETWEEN 4 PRECEDING  AND CURRENT ROW) AS sma_5,
        AVG(close) OVER (PARTITION BY ticker ORDER BY rn
                         ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS sma_21
      FROM base
    ),
    -- ...then compute the 5-day lag of SMA_21 in a separate step
    smas AS (
      SELECT
        *,
        LAG(sma_21, 5) OVER (PARTITION BY ticker ORDER BY rn) AS sma_21_lag5
      FROM smas0
    ),
    -- Close lags up to Kc using rn (no nested windows)
    lags AS (
      SELECT
        b.*,
        {close_lag_cols}
      FROM base b
      WINDOW w_rn AS (PARTITION BY ticker ORDER BY rn)
    ),
    -- Finite-window EMA12/EMA26 (normalize by available weights)
    emas AS (
      SELECT
        l.ticker, l.date, l.rn, l.close,
        ({ema12_num}) / NULLIF(({ema12_den}), 0) AS ema_12,
        ({ema26_num}) / NULLIF(({ema26_den}), 0) AS ema_26
      FROM lags l
    ),
    macd_base AS (
      SELECT
        e.ticker, e.date, e.rn, e.close, e.ema_12, e.ema_26,
        (e.ema_12 - e.ema_26) AS macd
      FROM emas e
    ),
    macd_lags AS (
      SELECT
        m.*,
        {macd_lag_cols}
      FROM macd_base m
      WINDOW w_rn AS (PARTITION BY ticker ORDER BY rn)
    ),
    macd_sig AS (
      SELECT
        ticker, date, rn, close, ema_12, ema_26, macd,
        ({sig_num}) / NULLIF(({sig_den}), 0) AS macd_signal
      FROM macd_lags
    )
    SELECT
      s.ticker,
      s.date,
      s.sma_5,
      s.sma_21,
      e.ema_12,
      e.ema_26,
      CASE WHEN s.sma_21 IS NULL OR s.sma_21 = 0 THEN 0
           ELSE (s.close - s.sma_21) / s.sma_21 END AS price_vs_sma_21,
      COALESCE(s.sma_21 - s.sma_21_lag5, 0) AS sma_21_slope,
      m.macd,
      m.macd_signal,
      (m.macd - m.macd_signal) AS macd_hist
    FROM smas s
    JOIN emas e     USING (ticker, rn)
    JOIN macd_sig m USING (ticker, rn)
    ORDER BY s.ticker, s.rn;
    """

    con.execute(sql)
    con.close()
    print("✅ feat_trend rebuilt (finite-window EMAs), no nested windows, 1 row per (ticker,date).")

if __name__ == "__main__":
    main()
