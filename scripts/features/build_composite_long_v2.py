#!/usr/bin/env python3
"""
build_composite_long_v2.py

Builds CL v2 (refined long-horizon composite) by combining:

    - quality_z, growth_z, inst_z   (from feat_composite_long)
    - value_z                       (from feat_value)
    - size_z, adv_z                 (from feat_adv)

Outputs:
    feat_composite_long_v2 with:
        ticker
        date
        quality_z
        growth_z
        inst_z
        value_z
        size_z
        adv_z
        alpha_CL_v2
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


def build_cl_v2(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building Composite Long v2 (CL v2)...")

    # Load CL v1
    cl1 = con.execute("""
        SELECT
            ticker,
            date,
            quality_z,
            growth_z,
            inst_z
        FROM feat_composite_long
    """).fetchdf()

    logger.info(f"Loaded {len(cl1):,} rows from feat_composite_long.")

    # Load value features
    val = con.execute("""
        SELECT
            ticker,
            date,
            value_z
        FROM feat_value
    """).fetchdf()

    logger.info(f"Loaded {len(val):,} rows from feat_value.")

    # Load ADV/size
    adv = con.execute("""
        SELECT
            ticker,
            date,
            adv_z,
            size_z
        FROM feat_adv
    """).fetchdf()

    logger.info(f"Loaded {len(adv):,} rows from feat_adv.")

    # Join all on (ticker, date)
    df = cl1.merge(val, on=["ticker", "date"], how="left")
    df = df.merge(adv, on=["ticker", "date"], how="left")

    logger.info(f"Joined CL v1 + value + adv: {len(df):,} rows total.")

    # Compute alpha_CL_v2 as mean of available z-scores
    components = ["quality_z", "growth_z", "inst_z", "value_z", "size_z", "adv_z"]

    df["alpha_CL_v2"] = df[components].mean(axis=1, skipna=True)

    out = df[["ticker", "date"] + components + ["alpha_CL_v2"]]

    con.execute("DROP TABLE IF EXISTS feat_composite_long_v2")
    con.register("df_cl2", out)
    con.execute("CREATE TABLE feat_composite_long_v2 AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM df_cl2")

    logger.info(f"feat_composite_long_v2 created with {len(out):,} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build Composite Long v2 (CL v2).")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_cl_v2(con)
    con.close()


if __name__ == "__main__":
    main()
