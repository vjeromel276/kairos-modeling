#!/usr/bin/env python3
"""
build_academic_composite_factors.py

Composite v2: price/volume + institutional

Builds a multi-factor composite using:

    - Trend:          price_vs_sma_21          (feat_trend)
    - Mean reversion: -close_zscore_21d        (feat_stat)
    - Volume surprise: vol_zscore_21d          (feat_volume_volatility)
    - Institutional:   inst_flow_value_qoq     (feat_institutional_academic)

All components are:
    - normalized cross-sectionally per date (z-scores)
    - aligned so "higher is better"

Outputs DuckDB table: feat_composite_academic

Columns:
    ticker
    date
    trend_z
    meanrev_z
    vol_z
    inst_z
    alpha_composite_eq    -- 3-factor equal-weight (price-based)
    alpha_composite_v2    -- 4-factor equal-weight (price+inst)
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
    """
    Compute z-score safely:
    - handles NaN
    - handles constant series (std=0)
    - returns 0 where we can't compute a valid z.
    """
    mean = series.mean()
    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - mean) / std


def build_composite(con: duckdb.DuckDBPyConnection) -> None:
    required = [
        "sep_base_academic",
        "feat_trend",
        "feat_stat",
        "feat_volume_volatility",
        "feat_institutional_academic",
    ]
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    missing = [t for t in required if t not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables for composite: {missing}")

    logger.info("Loading raw factor signals from DuckDB...")

    df = con.execute("""
        SELECT
            s.ticker,
            s.date,
            tr.price_vs_sma_21                    AS trend_raw,
            -st.close_zscore_21d                  AS meanrev_raw,
            vv.vol_zscore_21d                     AS vol_raw,
            inst.inst_flow_value_qoq              AS inst_raw
        FROM sep_base_academic s
        LEFT JOIN feat_trend tr
          USING (ticker, date)
        LEFT JOIN feat_stat st
          USING (ticker, date)
        LEFT JOIN feat_volume_volatility vv
          USING (ticker, date)
        LEFT JOIN feat_institutional_academic inst
          USING (ticker, date)
        ORDER BY s.date, s.ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for composite factor build.")

    logger.info("Computing per-date z-scores safely in pandas...")

    # trend, mean rev, volume, institutional all per-date z-scored
    df["trend_z"] = df.groupby("date")["trend_raw"].transform(safe_zscore)
    df["meanrev_z"] = df.groupby("date")["meanrev_raw"].transform(safe_zscore)
    df["vol_z"] = df.groupby("date")["vol_raw"].transform(safe_zscore)
    df["inst_z"] = df.groupby("date")["inst_raw"].transform(safe_zscore)

    logger.info("Computing composite alphas...")

    # Old 3-factor composite (price-only)
    df["alpha_composite_eq"] = df[["trend_z", "meanrev_z", "vol_z"]].mean(axis=1)

    # New v2 composite: add institutional z
    df["alpha_composite_v2"] = df[["trend_z", "meanrev_z", "vol_z", "inst_z"]].mean(axis=1)

    logger.info("Saving feat_composite_academic to DuckDB...")

    con.execute("DROP TABLE IF EXISTS feat_composite_academic")
    con.register("df_comp", df)
    con.execute("CREATE TABLE feat_composite_academic AS SELECT * FROM df_comp")

    logger.info(f"Composite factor table saved: {len(df):,} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build academic composite factor table (v2 with institutional).")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_composite(con)
    con.close()


if __name__ == "__main__":
    main()
