#!/usr/bin/env python3
"""
institutional_factor_academic.py

Builds an institutional factor table from SHARADAR SF3 holdings,
aligned with the academic base table sep_base_academic.

Assumes SF3 schema:

    ticker        VARCHAR
    investorname  VARCHAR
    securitytype  VARCHAR
    calendardate  DATE      -- quarter date (13F as-of date)
    value         DOUBLE    -- position value
    units         DOUBLE    -- position shares
    price         DOUBLE    -- price at report

Outputs:
    feat_institutional_academic (DuckDB table)

Columns (daily, per ticker):
    ticker
    date
    inst_value            - total reported value held by institutions (per quarter)
    inst_shares           - total reported shares held
    inst_num              - number of distinct institutions
    inst_flow_value_qoq   - QoQ % change in inst_value
    inst_flow_shares_qoq  - QoQ % change in inst_shares
    inst_num_change       - QoQ change in inst_num
"""

import argparse
import logging
import duckdb  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_institutional_factors(con: duckdb.DuckDBPyConnection) -> None:
    # ------------------------------------------------------------------
    # 1. Sanity checks
    # ------------------------------------------------------------------
    required_tables = ["sf3", "sep_base_academic"]
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    missing = [t for t in required_tables if t not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables for institutional factors: {missing}")

    logger.info("Building institutional factors from sf3 + sep_base_academic...")

    # Restrict to tickers that exist in sep_base_academic
    logger.info("Creating temp table of universe tickers from sep_base_academic...")
    con.execute("DROP TABLE IF EXISTS __inst_universe_tickers")
    con.execute("""
        CREATE TABLE __inst_universe_tickers AS
        SELECT DISTINCT ticker
        FROM sep_base_academic
    """)
    tickers_count = con.execute("SELECT COUNT(*) FROM __inst_universe_tickers").fetchone()[0]
    logger.info(f"Universe tickers in sep_base_academic: {tickers_count:,}")

    # ------------------------------------------------------------------
    # 2. Quarterly summary per ticker
    # ------------------------------------------------------------------
    logger.info("Aggregating sf3 to quarterly holdings per ticker...")

    con.execute("DROP TABLE IF EXISTS __inst_quarterly_raw")
    con.execute("""
        CREATE TABLE __inst_quarterly_raw AS
        WITH sf3_filtered AS (
            SELECT
                s.ticker,
                s.investorname,
                s.calendardate::DATE AS calendardate,
                s.value,
                s.units               AS shares
            FROM sf3 s
            JOIN __inst_universe_tickers u
              ON s.ticker = u.ticker
            WHERE s.calendardate IS NOT NULL
        ),
        by_quarter AS (
            SELECT
                ticker,
                DATE_TRUNC('quarter', calendardate)::DATE AS quarter_end,
                -- we don't have filingdate, so approximate "report date" as calendardate
                MAX(calendardate)::DATE AS report_date,
                SUM(value) AS tot_value,
                SUM(shares) AS tot_shares,
                COUNT(DISTINCT investorname) AS num_inst
            FROM sf3_filtered
            GROUP BY ticker, DATE_TRUNC('quarter', calendardate)
        ),
        with_lags AS (
            SELECT
                ticker,
                quarter_end,
                report_date,
                tot_value,
                tot_shares,
                num_inst,
                LAG(tot_value)  OVER (PARTITION BY ticker ORDER BY quarter_end) AS prev_value,
                LAG(tot_shares) OVER (PARTITION BY ticker ORDER BY quarter_end) AS prev_shares,
                LAG(num_inst)   OVER (PARTITION BY ticker ORDER BY quarter_end) AS prev_num_inst
            FROM by_quarter
        )
        SELECT
            ticker,
            quarter_end,
            report_date,
            tot_value,
            tot_shares,
            num_inst,
            CASE
                WHEN prev_value IS NULL OR prev_value = 0 THEN NULL
                ELSE (tot_value - prev_value) / prev_value
            END AS flow_value_qoq,
            CASE
                WHEN prev_shares IS NULL OR prev_shares = 0 THEN NULL
                ELSE (tot_shares - prev_shares) / prev_shares
            END AS flow_shares_qoq,
            (num_inst - prev_num_inst) AS num_inst_change
        FROM with_lags
    """)

    qrows = con.execute("SELECT COUNT(*) FROM __inst_quarterly_raw").fetchone()[0]
    logger.info(f"Quarterly institutional summary rows: {qrows:,}")

    # ------------------------------------------------------------------
    # 3. Effective date ranges (approximate 13F lag with 45 days)
    # ------------------------------------------------------------------
    logger.info("Computing effective date ranges with 45-day lag...")

    con.execute("DROP TABLE IF EXISTS __inst_quarterly_effective")
    con.execute("""
        CREATE TABLE __inst_quarterly_effective AS
        SELECT
            ticker,
            quarter_end,
            report_date,
            (report_date + INTERVAL 45 DAY)::DATE AS effective_from,
            LEAD((report_date + INTERVAL 45 DAY)::DATE)
                OVER (PARTITION BY ticker ORDER BY quarter_end) AS next_effective_from,
            tot_value,
            tot_shares,
            num_inst,
            flow_value_qoq,
            flow_shares_qoq,
            num_inst_change
        FROM __inst_quarterly_raw
    """)

    eff_rows = con.execute("SELECT COUNT(*) FROM __inst_quarterly_effective").fetchone()[0]
    logger.info(f"Effective quarterly rows: {eff_rows:,}")

    # ------------------------------------------------------------------
    # 4. Expand to daily factors using sep_base_academic
    # ------------------------------------------------------------------
    logger.info("Expanding institutional factors to daily using sep_base_academic...")

    con.execute("DROP TABLE IF EXISTS feat_institutional_academic")
    con.execute("""
        CREATE TABLE feat_institutional_academic AS
        SELECT
            d.ticker,
            d.date,
            q.tot_value        AS inst_value,
            q.tot_shares       AS inst_shares,
            q.num_inst         AS inst_num,
            q.flow_value_qoq   AS inst_flow_value_qoq,
            q.flow_shares_qoq  AS inst_flow_shares_qoq,
            q.num_inst_change  AS inst_num_change
        FROM __inst_quarterly_effective q
        JOIN sep_base_academic d
          ON d.ticker = q.ticker
         AND d.date >= q.effective_from
         AND (
             q.next_effective_from IS NULL
             OR d.date < q.next_effective_from
         )
    """)

    daily_rows = con.execute("SELECT COUNT(*) FROM feat_institutional_academic").fetchone()[0]
    logger.info(f"feat_institutional_academic created with {daily_rows:,} rows.")

    # Cleanup
    con.execute("DROP TABLE IF EXISTS __inst_universe_tickers")
    con.execute("DROP TABLE IF EXISTS __inst_quarterly_raw")
    con.execute("DROP TABLE IF EXISTS __inst_quarterly_effective")


def main():
    parser = argparse.ArgumentParser(
        description="Build institutional factor table from SHARADAR SF3."
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_institutional_factors(con)
    con.close()


if __name__ == "__main__":
    main()
