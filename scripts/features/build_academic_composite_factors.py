#!/usr/bin/env python3
"""
build_academic_composite_factors.py

FINAL VERSION — uses pandas for per-date z-scores,
avoids all DuckDB stddev failures.

Output:
    DuckDB table: feat_composite_academic
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def safe_zscore(series: pd.Series) -> pd.Series:
    """Compute z-score safely: handles NaNs, constant series, 0 std."""
    mean = series.mean()
    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - mean) / std


def build_composite(con: duckdb.DuckDBPyConnection):

    logger.info("Loading raw factor signals from DuckDB...")

    # Pull raw features from academic tables
    df = con.execute("""
        SELECT
            s.ticker,
            s.date,
            tr.price_vs_sma_21 AS trend_raw,
            -st.close_zscore_21d AS meanrev_raw,
            vv.vol_zscore_21d AS vol_raw
        FROM sep_base_academic s
        LEFT JOIN feat_trend tr USING (ticker, date)
        LEFT JOIN feat_stat st USING (ticker, date)
        LEFT JOIN feat_volume_volatility vv USING (ticker, date)
        ORDER BY s.date, s.ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for composite factor build.")

    # Compute z-scores per date safely
    logger.info("Computing per-date z-scores safely in pandas...")

    df["trend_z"] = df.groupby("date")["trend_raw"].transform(safe_zscore)
    df["meanrev_z"] = df.groupby("date")["meanrev_raw"].transform(safe_zscore)
    df["vol_z"] = df.groupby("date")["vol_raw"].transform(safe_zscore)

    logger.info("Computing composite factor...")

    df["alpha_composite_eq"] = df[["trend_z", "meanrev_z", "vol_z"]].mean(axis=1)

    # Save back to DuckDB
    logger.info("Saving feat_composite_academic to DuckDB...")

    con.execute("DROP TABLE IF EXISTS feat_composite_academic")
    con.register("df_comp", df)
    con.execute("CREATE TABLE feat_composite_academic AS SELECT * FROM df_comp")

    logger.info(f"Composite factor table saved: {len(df):,} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_composite(con)
    con.close()


if __name__ == "__main__":
    main()
