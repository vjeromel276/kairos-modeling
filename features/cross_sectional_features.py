# cross_sectional_features.py
# Computes cross-sectional percentile ranks and sector-neutral features

import argparse
import duckdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build cross-sectional features: percentile ranks and sector-neutral values"
    )
    parser.add_argument(
        "--db-path", default="data/kairos.duckdb",
        help="Path to DuckDB file containing feature tables and metadata"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = duckdb.connect(database=args.db_path)

    # Create or replace cross-sectional features table
    conn.execute("""
    CREATE OR REPLACE TABLE feat_cross_sectional AS
    WITH base AS (
      SELECT
        pa.ticker,
        pa.date,
        pa.ret_1d,
        pa.ret_5d,
        pa.ret_21d,
        sf.close_zscore_21d,
        tf.rsi_14,
        vol.vol_zscore_21d,
        tr.sma_5,
        fu.pe,
        fu.pb,
        m.sector
      FROM feat_price_action pa
      LEFT JOIN feat_stat sf      USING (ticker, date)
      LEFT JOIN feat_technical tf USING (ticker, date)
      LEFT JOIN feat_volume_volatility vol USING (ticker, date)
      LEFT JOIN feat_trend tr     USING (ticker, date)
      LEFT JOIN feat_fundamental fu USING (ticker, date)
      LEFT JOIN ticker_metadata_view m USING (ticker)
    )
    SELECT
      ticker,
      date,
      -- Percentile ranks across tickers each date
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY ret_1d)  AS cs_rank_ret1d,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY ret_5d)  AS cs_rank_ret5d,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY ret_21d) AS cs_rank_ret21d,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY close_zscore_21d) AS cs_rank_close_zscore,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY rsi_14)    AS cs_rank_rsi14,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY vol_zscore_21d) AS cs_rank_vol_zscore,
      PERCENT_RANK() OVER (PARTITION BY date ORDER BY sma_5)     AS cs_rank_sma5,
      -- Sector-neutralization: each feature minus the sector median for that date
      ret_5d - MEDIAN(ret_5d) OVER (PARTITION BY date, sector)                AS sec_neut_ret5d,
      close_zscore_21d - MEDIAN(close_zscore_21d) OVER (PARTITION BY date, sector) AS sec_neut_close_zscore,
      rsi_14 - MEDIAN(rsi_14) OVER (PARTITION BY date, sector)                AS sec_neut_rsi14
    FROM base;
    """)

    conn.close()
    print("✅ feat_cross_sectional built.")


if __name__ == "__main__":
    main()
