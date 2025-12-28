#!/usr/bin/env python3
"""
build_composite_long.py

Composite Long (CL) v1: slow-horizon factors

Components:
    - Quality:   roe, roa, grossmargin, netmargin  (from sf1 MRQ)
    - Growth:    revenue and net income growth QoQ (from sf1 MRQ)
    - Institutional: inst_flow_value_qoq          (from feat_institutional_academic)

Logic:
    1. Use SF1 with dimension='MRQ'
    2. Compute QoQ growth at the quarterly level
    3. Apply a 45-day lag on report date (datekey + 45 days)
    4. Expand fundamentals to daily using sep_base_academic
    5. Join in institutional factor
    6. Compute per-date z-scores (quality_z, growth_z, inst_z) in pandas
    7. Compute alpha_CL = (quality_z + growth_z + inst_z) / 3

Outputs:
    DuckDB table: feat_composite_long
"""

import argparse
import logging
import duckdb  # type: ignore
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def safe_zscore(series: pd.Series) -> pd.Series:
    """Compute z-score safely: handles NaNs and constant series."""
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def build_composite_long(con: duckdb.DuckDBPyConnection) -> None:
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    required = ["sf1", "sep_base_academic", "feat_institutional_academic"]
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    missing = [t for t in required if t not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables for Composite Long: {missing}")

    logger.info("Building Composite Long (CL) using SF1 MRQ + institutional factors...")

    # ------------------------------------------------------------------
    # 1. Universe tickers from sep_base_academic
    # ------------------------------------------------------------------
    logger.info("Creating universe ticker list from sep_base_academic...")
    con.execute("DROP TABLE IF EXISTS __cl_universe_tickers")
    con.execute("""
        CREATE TABLE __cl_universe_tickers AS
        SELECT DISTINCT ticker
        FROM sep_base_academic
    """)
    uni_count = con.execute("SELECT COUNT(*) FROM __cl_universe_tickers").fetchone()[0]
    logger.info(f"Universe tickers: {uni_count:,}")

    # ------------------------------------------------------------------
    # 2. Extract MRQ fundamentals and compute QoQ growth
    # ------------------------------------------------------------------
    logger.info("Aggregating SF1 MRQ fundamentals per quarter...")

    # NOTE: If any of the columns below (roe, roa, grossmargin, netmargin, revenue, netinc)
    # have slightly different names in your SF1, adjust them here.
    con.execute("DROP TABLE IF EXISTS __cl_quarterly_raw")
    con.execute("""
        CREATE TABLE __cl_quarterly_raw AS
        WITH sf1_mrq AS (
            SELECT
                s.ticker,
                s.datekey::DATE    AS report_date,
                s.roe,
                s.roa,
                s.grossmargin,
                s.netmargin,
                s.revenue,
                s.netinc
            FROM sf1 s
            JOIN __cl_universe_tickers u USING (ticker)
            WHERE s.dimension = 'MRQ'
              AND s.datekey IS NOT NULL
        ),
        with_lags AS (
            SELECT
                ticker,
                report_date,
                roe,
                roa,
                grossmargin,
                netmargin,
                revenue,
                netinc,
                LAG(revenue) OVER (PARTITION BY ticker ORDER BY report_date) AS prev_revenue,
                LAG(netinc)  OVER (PARTITION BY ticker ORDER BY report_date) AS prev_netinc
            FROM sf1_mrq
        )
        SELECT
            ticker,
            report_date,
            roe,
            roa,
            grossmargin,
            netmargin,
            revenue,
            netinc,
            CASE 
                WHEN prev_revenue IS NULL OR prev_revenue = 0 THEN NULL
                ELSE (revenue - prev_revenue) / prev_revenue
            END AS revenue_growth_qoq,
            CASE
                WHEN prev_netinc IS NULL OR prev_netinc = 0 THEN NULL
                ELSE (netinc - prev_netinc) / prev_netinc
            END AS netinc_growth_qoq
        FROM with_lags
    """)

    q_rows = con.execute("SELECT COUNT(*) FROM __cl_quarterly_raw").fetchone()[0]
    logger.info(f"Quarterly MRQ fundamentals rows: {q_rows:,}")

    # ------------------------------------------------------------------
    # 3. Compute effective date ranges (45-day lag)
    # ------------------------------------------------------------------
    logger.info("Computing effective date ranges (report_date + 45d lag)...")

    con.execute("DROP TABLE IF EXISTS __cl_quarterly_effective")
    con.execute("""
        CREATE TABLE __cl_quarterly_effective AS
        SELECT
            ticker,
            report_date,
            (report_date + INTERVAL 45 DAY)::DATE AS effective_from,
            LEAD((report_date + INTERVAL 45 DAY)::DATE)
                OVER (PARTITION BY ticker ORDER BY report_date) AS next_effective_from,
            roe,
            roa,
            grossmargin,
            netmargin,
            revenue_growth_qoq,
            netinc_growth_qoq
        FROM __cl_quarterly_raw
    """)

    eff_rows = con.execute("SELECT COUNT(*) FROM __cl_quarterly_effective").fetchone()[0]
    logger.info(f"Effective quarterly rows: {eff_rows:,}")

    # ------------------------------------------------------------------
    # 4. Expand to daily and join institutional factor
    # ------------------------------------------------------------------
    logger.info("Expanding CL factors to daily, joining feat_institutional_academic...")

    df = con.execute("""
        SELECT
            d.ticker,
            d.date,
            q.roe,
            q.roa,
            q.grossmargin,
            q.netmargin,
            q.revenue_growth_qoq,
            q.netinc_growth_qoq,
            inst.inst_flow_value_qoq
        FROM __cl_quarterly_effective q
        JOIN sep_base_academic d
          ON d.ticker = q.ticker
         AND d.date >= q.effective_from
         AND (
             q.next_effective_from IS NULL
             OR d.date < q.next_effective_from
         )
        LEFT JOIN feat_institutional_academic inst
          ON inst.ticker = d.ticker
         AND inst.date = d.date
        ORDER BY d.date, d.ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} daily rows for CL z-scoring.")

    # ------------------------------------------------------------------
    # 5. Compute per-date z-scores in pandas
    # ------------------------------------------------------------------
    logger.info("Computing quality_z, growth_z, inst_z in pandas...")

    # raw composites
    df["quality_raw"] = df[["roe", "roa", "grossmargin", "netmargin"]].mean(axis=1)
    df["growth_raw"] = df[["revenue_growth_qoq", "netinc_growth_qoq"]].mean(axis=1)
    df["inst_raw"] = df["inst_flow_value_qoq"]

    # z-scores by date
    df["quality_z"] = df.groupby("date")["quality_raw"].transform(safe_zscore)
    df["growth_z"] = df.groupby("date")["growth_raw"].transform(safe_zscore)
    df["inst_z"] = df.groupby("date")["inst_raw"].transform(safe_zscore)

    logger.info("Computing alpha_CL as equal-weight of quality_z, growth_z, inst_z...")

    df["alpha_CL"] = df[["quality_z", "growth_z", "inst_z"]].mean(axis=1)

    # ------------------------------------------------------------------
    # 6. Save to DuckDB as feat_composite_long
    # ------------------------------------------------------------------
    logger.info("Saving feat_composite_long to DuckDB...")

    con.execute("DROP TABLE IF EXISTS feat_composite_long")
    con.register("df_cl", df)
    con.execute("""
        CREATE TABLE feat_composite_long AS
        SELECT * FROM df_cl
    """)

    logger.info(f"feat_composite_long created with {len(df):,} rows.")

    # Cleanup temp tables
    con.execute("DROP TABLE IF EXISTS __cl_universe_tickers")
    con.execute("DROP TABLE IF EXISTS __cl_quarterly_raw")
    con.execute("DROP TABLE IF EXISTS __cl_quarterly_effective")


def main():
    parser = argparse.ArgumentParser(description="Build Composite Long (CL) v1 from SF1 MRQ + institutional.")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_composite_long(con)
    con.close()


if __name__ == "__main__":
    main()
