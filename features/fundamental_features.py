# features/fundamental_features.py
# Build fundamentals (point-in-time safe) from sf1 + daily with full coverage.

import argparse
import duckdb

def parse_args():
    p = argparse.ArgumentParser(description="Build fundamental features from SF1 (Sharadar) + daily")
    p.add_argument("--db-path", default="data/kairos.duckdb",
                   help="Path to DuckDB database")
    p.add_argument("--sf1-dimension", default="ART",
                   help="SF1 dimension to use (e.g., ART=TTM, ARQ=quarterly, MRY=annual)")
    return p.parse_args()

def main():
    args = parse_args()
    con = duckdb.connect(database=args.db_path)

    con.execute(f"""
    CREATE OR REPLACE TABLE feat_fundamental AS
    WITH
    -- 1) Pull TTM (or chosen dimension) rows from sf1; compute YoY on report rows
    m_raw AS (
      SELECT
        ticker,
        CAST(datekey AS DATE)     AS report_date,   -- filing date (as-of boundary)
        CAST(calendardate AS DATE) AS period_end,   -- period end (not used for as-of)
        eps, bvps, sps, ebitda, revenueusd
      FROM sf1
      WHERE dimension = '{args.sf1_dimension}'
    ),
    m_enriched AS (
      SELECT
        ticker,
        report_date,
        period_end,
        eps, bvps, sps, ebitda, revenueusd,
        (eps / NULLIF(LAG(eps, 4) OVER (PARTITION BY ticker ORDER BY report_date), 0)) - 1
          AS eps_yoy_report,
        (revenueusd / NULLIF(LAG(revenueusd, 4) OVER (PARTITION BY ticker ORDER BY report_date), 0)) - 1
          AS rev_yoy_report
      FROM m_raw
    ),

    -- 2) For each (ticker, trade day), pick latest report with report_date <= trade date
    asof_pick AS (
      SELECT
        s.ticker,
        CAST(s.date AS DATE) AS date,
        s.close,
        mv.sector,
        m.eps, m.bvps, m.sps, m.ebitda, m.revenueusd,
        m.eps_yoy_report AS eps_yoy,
        m.rev_yoy_report AS rev_yoy,
        ROW_NUMBER() OVER (
          PARTITION BY s.ticker, s.date
          ORDER BY m.report_date DESC
        ) AS rn
      FROM sep_base_common s
      LEFT JOIN m_enriched m
        ON m.ticker = s.ticker
       AND m.report_date <= s.date
      LEFT JOIN ticker_metadata_view mv
        ON mv.ticker = s.ticker
    ),
    asof_final AS (
      SELECT
        ticker, date, close,
        COALESCE(sector, 'Unknown') AS sector,
        eps, bvps, sps, ebitda, revenueusd,
        eps_yoy, rev_yoy
      FROM asof_pick
      WHERE rn = 1 OR rn IS NULL  -- keep rows even if no report exists yet
    ),

    -- 3) Fill missing raw metrics via sector-day → market-day → global → 0
    filled_raw AS (
      SELECT
        ticker, date, close, sector,
        COALESCE(
          eps,
          MEDIAN(eps) OVER (PARTITION BY date, sector),
          MEDIAN(eps) OVER (PARTITION BY date),
          MEDIAN(eps) OVER (),
          0
        ) AS eps_eff,
        COALESCE(
          bvps,
          MEDIAN(bvps) OVER (PARTITION BY date, sector),
          MEDIAN(bvps) OVER (PARTITION BY date),
          MEDIAN(bvps) OVER (),
          0
        ) AS bvps_eff,
        COALESCE(
          sps,
          MEDIAN(sps) OVER (PARTITION BY date, sector),
          MEDIAN(sps) OVER (PARTITION BY date),
          MEDIAN(sps) OVER (),
          0
        ) AS sps_eff,
        COALESCE(
          ebitda,
          MEDIAN(ebitda) OVER (PARTITION BY date, sector),
          MEDIAN(ebitda) OVER (PARTITION BY date),
          MEDIAN(ebitda) OVER (),
          0
        ) AS ebitda_eff,
        COALESCE(
          revenueusd,
          MEDIAN(revenueusd) OVER (PARTITION BY date, sector),
          MEDIAN(revenueusd) OVER (PARTITION BY date),
          MEDIAN(revenueusd) OVER (),
          0
        ) AS revenue_eff,
        COALESCE(
          eps_yoy,
          MEDIAN(eps_yoy) OVER (PARTITION BY date, sector),
          MEDIAN(eps_yoy) OVER (PARTITION BY date),
          MEDIAN(eps_yoy) OVER (),
          0
        ) AS eps_yoy_eff,
        COALESCE(
          rev_yoy,
          MEDIAN(rev_yoy) OVER (PARTITION BY date, sector),
          MEDIAN(rev_yoy) OVER (PARTITION BY date),
          MEDIAN(rev_yoy) OVER (),
          0
        ) AS rev_yoy_eff
      FROM asof_final
    ),

    -- 4) Bring in daily EV (exact trade-day value)
    with_ev AS (
      SELECT
        fr.*,
        d.ev AS ev_daily
      FROM filled_raw fr
      LEFT JOIN daily d
        ON d.ticker = fr.ticker AND d.date = fr.date
    ),

    -- 5) Compute valuations; fill EV/EBITDA after the fact
    valuations AS (
      SELECT
        ticker,
        date,
        -- ratios from price and per-share metrics
        CASE WHEN eps_eff  = 0 THEN 0 ELSE close / eps_eff END AS pe,
        CASE WHEN bvps_eff = 0 THEN 0 ELSE close / bvps_eff END AS pb,
        CASE WHEN sps_eff  = 0 THEN 0 ELSE close / sps_eff END AS ps,
        CASE WHEN ebitda_eff = 0 OR ev_daily IS NULL THEN NULL
             ELSE (ev_daily / ebitda_eff) END AS ev_ebitda_raw,
        -- carry cleaned fundamentals
        eps_eff    AS eps,
        revenue_eff AS revenue,
        eps_yoy_eff AS eps_yoy,
        rev_yoy_eff AS rev_yoy,
        sector
      FROM with_ev
    ),

    -- 6) Fill EV/EBITDA with sector/market/global medians (then 0)
    final_fill AS (
      SELECT
        ticker, date,
        pe, pb, ps,
        COALESCE(
          ev_ebitda_raw,
          MEDIAN(ev_ebitda_raw) OVER (PARTITION BY date, sector),
          MEDIAN(ev_ebitda_raw) OVER (PARTITION BY date),
          MEDIAN(ev_ebitda_raw) OVER (),
          0
        ) AS ev_ebitda,
        eps, revenue, eps_yoy, rev_yoy
      FROM valuations
    )

    SELECT
      ticker, date,
      pe, pb, ps, ev_ebitda,
      eps, revenue, eps_yoy, rev_yoy
    FROM final_fill
    ORDER BY ticker, date;
    """)

    con.close()
    print("✅ feat_fundamental built (sf1 as-of by filing date + daily EV + full coverage).")

if __name__ == "__main__":
    main()
