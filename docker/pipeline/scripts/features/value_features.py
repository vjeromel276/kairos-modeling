#!/usr/bin/env python3
"""
value_features.py

Builds a simple value factor from SF1 MRQ fundamentals and METRICS price:

    value_raw = -(price / tbvps)

Expanded daily via effective_from windows like CL v1.

Outputs:
    feat_value (DuckDB table)
        ticker
        date
        value_raw
        value_z
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def safe_z(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def build_value(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building value features from sf1 MRQ + metrics price...")

    # 1) MRQ tbvps per quarter
    con.execute("DROP TABLE IF EXISTS __val_quarterly_raw")
    con.execute("""
        CREATE TABLE __val_quarterly_raw AS
        SELECT
            ticker,
            datekey::DATE AS report_date,
            tbvps
        FROM sf1
        WHERE dimension = 'MRQ'
          AND tbvps IS NOT NULL
          AND datekey IS NOT NULL
    """)

    rows = con.execute("SELECT COUNT(*) FROM __val_quarterly_raw").fetchone()[0]
    logger.info(f"Quarterly MRQ tbvps rows: {rows:,}")

    # 2) Effective-from windows (report_date + 45 days)
    con.execute("DROP TABLE IF EXISTS __val_quarterly_effective")
    con.execute("""
        CREATE TABLE __val_quarterly_effective AS
        SELECT
            ticker,
            report_date,
            (report_date + INTERVAL 45 DAY)::DATE AS effective_from,
            LEAD((report_date + INTERVAL 45 DAY)::DATE)
                OVER (PARTITION BY ticker ORDER BY report_date) AS next_effective_from,
            tbvps
        FROM __val_quarterly_raw
    """)

    # 3) Expand to daily with metrics price
    df = con.execute("""
        SELECT
            m.ticker,
            m.date,
            m.price,
            q.tbvps
        FROM __val_quarterly_effective q
        JOIN metrics m
          ON m.ticker = q.ticker
         AND m.date >= q.effective_from
         AND (q.next_effective_from IS NULL OR m.date < q.next_effective_from)
        WHERE m.price IS NOT NULL
          AND q.tbvps IS NOT NULL
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for daily value factor.")

    df = df[df["tbvps"] > 0]
    df["price_to_tbv"] = df["price"] / df["tbvps"]
    df["value_raw"] = -df["price_to_tbv"]  # cheaper → more positive value_raw

    # Cross-sectional z-score per date
    df["value_z"] = df.groupby("date")["value_raw"].transform(safe_z)

    out = df[["ticker", "date", "value_raw", "value_z"]]

    con.execute("DROP TABLE IF EXISTS feat_value")
    con.register("df_val", out)
    con.execute("CREATE TABLE feat_value AS SELECT * FROM df_val")

    logger.info(f"feat_value created with {len(out):,} rows.")

    # Cleanup temp tables
    con.execute("DROP TABLE IF EXISTS __val_quarterly_raw")
    con.execute("DROP TABLE IF EXISTS __val_quarterly_effective")


def main():
    parser = argparse.ArgumentParser(description="Build value factor features.")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_value(con)
    con.close()


if __name__ == "__main__":
    main()
