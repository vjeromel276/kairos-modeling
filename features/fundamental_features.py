# fundamental_features.py
# Computes fundamental and valuation ratios from Sharadar metrics/fundamentals

import argparse
import duckdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build fundamental features from Sharadar metrics"
    )
    parser.add_argument(
        "--db-path", default="data/kairos.duckdb",
        help="Path to DuckDB file containing sep_base and metrics tables"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = duckdb.connect(database=args.db_path)

    # Create or replace fundamental features table
    conn.execute("""
    CREATE OR REPLACE TABLE feat_fundamental AS
    WITH metrics_lag AS (
      SELECT
        m.ticker,
        s.date AS trade_date,
        -- select all metrics up to the trade date
        LAST_VALUE(m.pe)    OVER w AS pe,
        LAST_VALUE(m.pb)    OVER w AS pb,
        LAST_VALUE(m.ps)    OVER w AS ps,
        LAST_VALUE(m.ev_to_ebitda) OVER w AS ev_ebitda,
        LAST_VALUE(m.eps)   OVER w AS eps,
        LAST_VALUE(m.revenue) OVER w AS revenue,
        -- year-over-year growth: compare to 4 quarters ago
        (LAST_VALUE(m.revenue) OVER w)
          / NULLIF(LAG(LAST_VALUE(m.revenue) OVER w, 4) OVER (PARTITION BY m.ticker ORDER BY s.date),0) - 1
          AS rev_yoy,
        (LAST_VALUE(m.eps) OVER w)
          / NULLIF(LAG(LAST_VALUE(m.eps) OVER w, 4) OVER (PARTITION BY m.ticker ORDER BY s.date),0) - 1
          AS eps_yoy
      FROM sep_base s
      LEFT JOIN sharadar_metrics m
        ON s.ticker = m.ticker
       AND m.date <= s.date
      WINDOW w AS (
        PARTITION BY s.ticker, s.date
        ORDER BY m.date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      )
    )
    SELECT DISTINCT
      ticker,
      trade_date AS date,
      pe,
      pb,
      ps,
      ev_ebitda,
      eps,
      revenue,
      rev_yoy,
      eps_yoy
    FROM metrics_lag;
    """)

    conn.close()
    print("✅ feat_fundamental built.")

if __name__ == "__main__":
    main()
