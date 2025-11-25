#!/usr/bin/env python3
"""
adv_features.py

Builds ADV (average dollar volume) and a simple size/liquidity factor
from sep_base_academic.

Outputs:
    feat_adv (DuckDB table) with:
        ticker
        date
        dollar_volume
        adv_20
        adv_60
        adv_z
        size_raw
        size_z
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


def build_adv(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building ADV and size features from sep_base_academic...")

    # Pull minimal price/volume from sep_base_academic
    df = con.execute("""
        SELECT
            ticker,
            date,
            close,
            volume
        FROM sep_base_academic
        ORDER BY ticker, date
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows from sep_base_academic for ADV calc.")

    df["dollar_volume"] = df["close"] * df["volume"]

    # Compute rolling ADV_20, ADV_60 per ticker
    def add_adv(group):
        group["adv_20"] = group["dollar_volume"].rolling(20).mean()
        group["adv_60"] = group["dollar_volume"].rolling(60).mean()
        return group

    df = df.groupby("ticker", group_keys=False).apply(add_adv)

    # Size proxy: long-term average dollar volume per ticker
    size_map = df.groupby("ticker")["dollar_volume"].mean().rename("size_raw")
    df = df.join(size_map, on="ticker")

    # ADV z-score cross-sectionally per date
    df["adv_z"] = df.groupby("date")["adv_20"].transform(safe_z)

    # Size z-score cross-sectionally per date
    df["size_z"] = df.groupby("date")["size_raw"].transform(safe_z)

    # Drop na from early rolling windows
    df = df.dropna(subset=["adv_20", "adv_60"])

    out = df[[
        "ticker",
        "date",
        "dollar_volume",
        "adv_20",
        "adv_60",
        "adv_z",
        "size_raw",
        "size_z",
    ]]

    con.execute("DROP TABLE IF EXISTS feat_adv")
    con.register("df_adv", out)
    con.execute("CREATE TABLE feat_adv AS SELECT * FROM df_adv")

    logger.info(f"feat_adv created with {len(out):,} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build ADV and size features.")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_adv(con)
    con.close()


if __name__ == "__main__":
    main()
