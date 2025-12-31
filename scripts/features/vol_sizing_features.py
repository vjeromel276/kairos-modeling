#!/usr/bin/env python3
"""
vol_sizing_features.py

Builds volatility measures for position sizing:
    - vol_21: 21-day realized volatility
    - vol_63: 63-day realized volatility
    - vol_blend: 0.5 * vol_21 + 0.5 * vol_63

Source:
    sep_base_academic

Output:
    feat_vol_sizing
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


def build_vol_sizing(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Loading sep_base_academic for vol sizing...")
    df = con.execute("""
        SELECT
            ticker,
            date,
            close
        FROM sep_base_academic
        ORDER BY ticker, date
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for vol sizing.")

    # Compute daily returns per ticker
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    def add_vol(group: pd.DataFrame) -> pd.DataFrame:
        # Rolling realized vol (std of daily returns)
        group["vol_21"] = group["ret_1d"].rolling(21).std()
        group["vol_63"] = group["ret_1d"].rolling(63).std()
        return group

    df = df.groupby("ticker", group_keys=False).apply(add_vol)

    # Blended vol
    df["vol_blend"] = 0.5 * df["vol_21"] + 0.5 * df["vol_63"]

    # Drop rows without full history
    df = df.dropna(subset=["vol_21", "vol_63", "vol_blend"])

    out = df[["ticker", "date", "vol_21", "vol_63", "vol_blend"]]

    con.execute("DROP TABLE IF EXISTS feat_vol_sizing")
    con.register("df_vol", out)
    con.execute("CREATE TABLE feat_vol_sizing AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM df_vol")

    logger.info(f"feat_vol_sizing created with {len(out):,} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build vol sizing features.")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_vol_sizing(con)
    con.close()


if __name__ == "__main__":
    main()
